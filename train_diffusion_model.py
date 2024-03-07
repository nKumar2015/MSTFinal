# based on https://github.com/huggingface/diffusers/blob/main/examples/train_unconditional.py

import os
import pickle
from typing import Optional

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from datasets import load_dataset
from diffusers import (DDPMScheduler, UNet2DConditionModel)
from diffusers.optimization import get_scheduler
from Mel import Mel
from diffusers import EMAModel
from huggingface_hub import HfFolder, Repository, whoami
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm.auto import tqdm

from audio_pipline import AudioDiffusionPipeline

config = {
    'dataset': 'Nkumar5/RockMST', # Repo name of dataset on Huggingface Hub
    'hub_model_id':'NKumar5/CAGRock', # Repo name of model on Huggingface Hub
    'output_dir': 'results/RockMST',
    'epochs': 1000, 
    'batch_size': 2, # Change according to available VRAM. 2 works best on 24GB VRAM
    'eval_batch_size': 2, # How many image to test on at once
    'grad_accumulation_steps': 8, # To increase effective batch // size see https://huggingface.co/docs/accelerate/usage_guides/gradient_accumulation
    'learning_rate': 1e-4,
    'warmup_steps': 500,
    'mixed_precision': 'bf16', # 'bf16', 'fp16', or None
    'push_to_hub': True, # Pushes model to huggingface hub. Must be logged into Huggingface CLI
    'encodings': 'data/rock_encodings.p',
    'save_model_epochs': 10,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)
print(f"Using {device} Device")

logger = get_logger(__name__)

def get_full_repo_name(model_id: str,
                       organization: Optional[str] = None,
                       token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)['name']
        return f'{username}/{model_id}'
    else:
        return f'{organization}/{model_id}'
    

output_dir = os.environ.get('SM_MODEL_DIR', None) or config['output_dir']
accelerator = Accelerator(
    gradient_accumulation_steps=config['grad_accumulation_steps'],
    mixed_precision=config['mixed_precision'],
)

dataset = load_dataset(config['dataset'], use_auth_token=True, split="train")

resolution = dataset[0]['image'].height, dataset[0]['image'].width

augmentations = Compose([
    ToTensor(),
    Normalize(mean=[0.5], std=[0.5]),
])

def transforms(examples):
    images = [augmentations(images) for images in examples['image']]
    encoding = [encodings[file] for file in examples["audio_file"]]
    return {"input": images, "encoding": encoding}

dataset.set_transform(transforms)
train_dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=config['batch_size'], shuffle=True, generator=torch.Generator(device=device)
)

encodings = pickle.load(open(config['encodings'], 'rb'))

unet = UNet2DConditionModel(
    sample_size=resolution,
    in_channels=1,
    out_channels=1,
    layers_per_block=2,
    block_out_channels=(128, 256, 512, 512),
    down_block_types=(
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
        "DownBlock2D"
    ),
    up_block_types=(
        "UpBlock2D",
        "CrossAttnUpBlock2D",
        "CrossAttnUpBlock2D",
        "CrossAttnUpBlock2D",
    ),
    cross_attention_dim=list(encodings.values())[0].shape[-1]
)


noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

optimizer = torch.optim.Adam(
    unet.parameters(),
    lr=config['learning_rate'],
    betas=(0.95, 0.999),
)

learning_rate_scheduler = get_scheduler(
    "cosine",
    optimizer=optimizer,
    num_warmup_steps=config['warmup_steps'],
    num_training_steps=(len(train_dataloader) * config['epochs']) // config['grad_accumulation_steps'],
)

unet, optimizer, train_dataloader, learning_rate_scheduler = accelerator.prepare(
    unet, optimizer, train_dataloader, learning_rate_scheduler)

ema_model = EMAModel(
    getattr(unet, "module", unet),
    inv_gamma=1.0,
    power=3/4,
    max_valiue=0.9999
)

repo = Repository(output_dir, clone_from=config['hub_model_id'])

if accelerator.is_main_process:
    run = os.path.split(__file__)[-1].split(".")[0]
    accelerator.init_trackers(run)

mel = Mel(
    x_res=resolution[1],
    y_res=resolution[0],
    hop_length=512,
    sample_rate=22050,
    n_fft=2048
)

global_step = 0

for epoch in range(config['epochs']):
    progress_bar = tqdm(total=len(train_dataloader),
                        disable=not accelerator.is_local_main_process)
    progress_bar.set_description(f"Epoch #{epoch}")

    unet.train()

    for step, batch in enumerate(train_dataloader):
        clean_images = batch['input'].to(device)

        noise = torch.randn(clean_images.shape)
        bsz = clean_images.shape[0]

        timesteps = torch.randint(
            0,
            noise_scheduler.config.num_train_timesteps,
            (bsz, ),
            device=device,
        ).long()

        noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

        with accelerator.accumulate(unet):
            noise_pred = unet(noisy_images, timesteps, batch['encoding'].to(device))['sample']
            loss = F.mse_loss(noise_pred, noise)
            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(unet.parameters(), 1.0)
            
            optimizer.step()
            learning_rate_scheduler.step()
            ema_model.step(unet)
            optimizer.zero_grad()

        progress_bar.update(1)
        global_step += 1

        logs = {
            "loss": loss.detach().item(),
            "lr": learning_rate_scheduler.get_last_lr()[0],
            "ema_decay": ema_model.decay,
            "step": global_step
        }

        progress_bar.set_postfix(**logs)
        accelerator.log(logs, step=global_step)
    
    progress_bar.close()

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        if((epoch + 1) % config['save_model_epochs'] == 0 
           or (epoch + 1) == config['epochs']):
            
            unet = accelerator.unwrap_model(unet)
            ema_model.copy_to(unet.parameters())
            pipline = AudioDiffusionPipeline(vqvae=None, unet=unet, mel=mel, scheduler=noise_scheduler)

            pipline.save_pretrained(output_dir)
            repo.push_to_hub(
                commit_message=f'Epoch {epoch}',
                blocking=True,
                auto_lfs_prune=True,
            )

accelerator.end_training()