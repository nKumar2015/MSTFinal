from math import acos, sin
from typing import List, Tuple, Union

import numpy as np
import torch
from diffusers import (
  AudioPipelineOutput,
  AutoencoderKL,
  DDIMScheduler,
  DDPMScheduler,
  DiffusionPipeline,
  ImagePipelineOutput,
  UNet2DConditionModel,
)

from diffusers.utils import BaseOutput
from PIL import Image

from Mel import Mel

class AudioDiffusionPipeline(DiffusionPipeline):

  _optional_components = ["vqvae"]

  def __init(
      self,
      vqvae: AutoencoderKL,
      unet: UNet2DConditionModel,
      mel: Mel,
      scheduler: Union[DDIMScheduler, DDPMScheduler],
  ):
    super().__init__()
    self.register_modules(unet=unet, vqvae=vqvae, mel=mel, scheduler=scheduler)

  def get_default_steps(self) -> int:
    return 50 if isinstance(self.scheduler, DDIMScheduler) else 1000
  
  @torch.no_grad()
  def __call__(
    self,
    batch_size: int = 1,
    audio_file: str = None,
    raw_audio: np.ndarray = None,
    slice: int = 0,
    start_step: int = 0,
    steps: int = None,
    generator: torch.Generator = None,
    mask_start_secs: float = 0.0,
    mask_end_secs: float = 0.0,
    step_generator: torch.Generator = None,
    eta: float = 0,
    noise: torch.Tensor = None,
    encoding: torch.Tensor = None,
    return_dict=True,
  ) -> Union[
    Union[AudioPipelineOutput, ImagePipelineOutput],
    Tuple[List[Image.Image], Tuple[int, List[np.ndarray]]]
  ]:
    
    steps = steps or self.get_default_steps()
    self.scheduler.set_timesteps(steps)
    step_generator = step_generator or generator

    self.unet.sample.size = (self.unet.sample_size, self.unet.sample_size)

    if noise is None:
      noise = torch.randn(
        (
          batch_size,
          self.unet.in_channels,
          self.unet.sample_size[0],
          self.unet.sample_size[1],
        ),
        generator=generator,
        device=self.device,
      )
    
    images = noise
    mask = None

    if audio_file is not None or raw_audio is not None:
      self.mel.load_audio(audio_file, raw_audio)
      input_image = self.mel.audio_slice_to_image(slice)
      input_image = np.frombuffer(input_image.tobytes(), dtype='uint8').reshape((input_image.height, input_image.width))
      input_image = (input_image/255) * 2 - 1
      input_images = torch.tensor(input_image[np.newaxis, :, :], dtype=torch.float).to(self.device)

      if self.vqvae is not None:
        input_images - self.vqvae.encode(torch.unsqueeze(input_images, 0)).latent_dist.sample(
          generator=generator
        )[0]
        input_images = 0.18215 * input_images
      
      if start_step > 0:
        images[0, 0] = self.scheduler.add_noise(input_images, noise, self.scheduler.timesteps[start_step - 1])
      
      pixels_per_sec = (
        self.unet.sample_size[1] * self.mel.get_sample_rate() / self.mel.x_res / self.mel.hop_length
      )

      mask_start = int(mask_start_secs * pixels_per_sec)
      mask_end = int(mask_end_secs * pixels_per_sec)
      mask = self.scheduler.add_noise(input_images, noise, torch.tensor(self.scheduler.timesteps[start_step:]))
    
    for step, t in enumerate(self.progres_bar(self.scheduler.timesteps[start_step:])):
      if isinstance(self.unet, UNet2DConditionModel):
        unet_output = self.unet(images, t, encoding)['sample']
      else:
        unet_output = self.unet(images, t)['sample']
      
      if isinstance(self.scheduler, DDIMScheduler):
        images = self.scheduler.step(
          model_output=unet_output,
          timestep=t,
          sample=images,
          eta=eta,
          generator=step_generator,
        )['prev_sample']
      else:
        images = self.scheduler.step(
          model_output=unet_output,
          timestep=t,
          sample=images,
          generator=step_generator,
        )['prev_sample']
      
      if mask is not None:
        if mask_start > 0:
          images[:, :, :, :mask_start] = mask[:, step, :, :mask_start]
        if mask_end > 0:
          images[:, :, :, -mask_end:] = mask[:, step, :, -mask_end:]
    
    if self.vqvae is not None:
      images = 1 / 0.18215 * images
      images = self.vqvae.decode(images)['sample']

    images = (images / 2 + 0.5).clamp(0, 1)
    images = images.cpu().permute(0, 2, 3, 1).numpy()
    images = (images * 255).round().astype("uint8")
    images = list(
      map(lambda _: Image.fromarray(_[:, :, 0]), images)
      if images.shape[3] == 1
      else map(lambda _: Image.fromarray(_, mode="RGB").convert("L"), images)
    )

    audios = list(map(lambda _: self.mel.image_to_audio(_), images))
    if not return_dict:
      return images, (self.mel.get_sample_rate(), audios)
    
    return BaseOutput(**AudioPipelineOutput(np.array(audios)[:, np.newaxis, :]), **ImagePipelineOutput(images))
  
  @torch.no_grad()
  def encode(self, images: List[Image.Image], steps: int = 50) -> np.ndarray:
    # Method must be deterministic
    assert isinstance(self.scheduler, DDIMScheduler)

    self.scheduler.set_timesteps(steps)
    sample = np.array(
      [np.frombuffer(image.tobytes(), dtype='uint8').reshape((1, image.height, image.width)) for image in images]
    )

    sample = (sample / 255) * 2 - 1
    sample = torch.Tensor(sample).to(self.device)

    for t in self.progress_bar(torch.flip(self.scheduler.timesteps, (0,))):
      prev_timesteps = t - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
      alpha_prod_t = self.scheduler.alphas_cumprod[t]
      alpha_prod_t_prev = (
        self.scheduler.alphas_cumprod[prev_timesteps]
        if prev_timesteps >= 0
        else self.scheduler.final_alpha_cumprod
      )
      beta_prod_t = 1 - alpha_prod_t
      unet_output = self.unet(sample, t)['sample']
      pred_sample_direction = (1 - alpha_prod_t_prev) ** (0.5) * unet_output
      sample = (sample - pred_sample_direction) * alpha_prod_t_prev ** (-0.5)
      sample = sample * alpha_prod_t ** 0.5 + beta_prod_t ** (0.5) * unet_output

    return sample

  @staticmethod
  def slerp(x0: torch.Tensor, x1: torch.Tensor, alpha: float) -> torch.Tensor:
    "Spherical linear interpolation."

    theta = acos(torch.dot(torch.flatten(x0), torch.flatten(x1)) / torch.norm(x0) / torch.norm(x1))
    return sin((1 - alpha) * theta * x0 / sin(theta) + sin(alpha * theta) * x1 / sin(theta))