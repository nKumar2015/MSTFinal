from datasets import load_dataset

dataset = load_dataset("Nkumar5/FMARock", use_auth_token=True, split="train")

import matplotlib.pyplot as plt
import torch
from PIL import Image
from diffusers import DDPMScheduler
from torchvision import transforms

preprocess = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)

def transform(examples):
    images = [preprocess(image.convert("RGB")) for image in examples["image"]]
    return {"images": images}

dataset.set_transform(transform)

fig, axs = plt.subplots(1, 4, figsize=(16, 4))


for i, spectrogram in enumerate(dataset[:4]['images']):
  spectrogram = spectrogram.unsqueeze(0)
  spec = Image.fromarray(((spectrogram.permute(0, 2, 3, 1) + 1.0) * 127.5).type(torch.uint8).numpy()[0])

  axs[i].imshow(spec)
  axs[i].set_axis_off()

plt.show()

sample_image = dataset[0]["images"].unsqueeze(0)

noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
noise = torch.randn(sample_image.shape)
timesteps = torch.LongTensor([1])

fig2, axs2 = plt.subplots(1, 4, figsize=(16, 4))

for i, spectrogram in enumerate(dataset[:4]['images']):
  spectrogram = spectrogram.unsqueeze(0)
  noisy_image = noise_scheduler.add_noise(spectrogram, noise, timesteps)
  image = Image.fromarray(((noisy_image.permute(0, 2, 3, 1) + 1.0) * 127.5).type(torch.uint8).numpy()[0])

  axs2[i].imshow(image)
  axs2[i].set_axis_off()

plt.show()