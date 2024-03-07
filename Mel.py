import warnings
from typing import Callable, Union

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin

warnings.filterwarnings("ignore")

import numpy as np
import librosa
from PIL import Image

class Mel(ConfigMixin, SchedulerMixin):
  config_name = "mel_config.json"

  @register_to_config
  def __init__(
    self,
    x_res: int = 256,
    y_res: int = 256,
    sample_rate: int = 22050,
    n_fft: int = 2048,
    hop_length: int = 512,
    top_db: float = 80.0,
    n_iter: int = 32,
    ):
      self.hop_length = hop_length
      self.sr = sample_rate
      self.n_fft = n_fft
      self.top_db = top_db
      self.audio = None
      self.n_iter = n_iter
      self.set_resolution(x_res, y_res)
    
  def set_resolution(self, x_res: int, y_res: int):
    self.x_res = x_res
    self.y_res = y_res
    self.n_mels = self.y_res
    self.slice_size = self.x_res * self.hop_length - 1
    
  def load_audio(self, audio_file: str = None, raw_audio: np.ndarray = None):
    if audio_file is not None:
      self.audio, _ = librosa.load(audio_file, mono=True, sr=self.sr)
    else:
      self.audio = raw_audio
    
    if len(self.audio) < self.x_res * self.hop_length:
      self.audio = np.concatenate([self.audio, np.zeros((self.x_res * self.hop_length - len(self.audio),))])

  def get_number_of_slices(self) -> int:
    return len(self.audio) // self.slice_size

  def get_audio_slice(self, slice: int = 0) -> int:
    return self.audio[self.slice_size * slice : self.slice_size * (slice + 1)]
  
  def get_sample_rate(self) -> int:
    return self.sr

  def audio_slice_to_image(self, slice: int, ref: Union[float, Callable] = np.max) -> Image.Image:
    S = librosa.feature.melspectrogram(
      y=self.get_audio_slice(slice),
      sr=self.sr,
      n_fft=self.n_fft,
      hop_length=self.hop_length,
      n_mels=self.n_mels,
    )
    log_S = librosa.power_to_db(S, ref=ref, top_db=self.top_db)
    spec_data = (((log_S + self.top_db) * 255 / self.top_db).clip(0, 255) + 0.5).astype(np.uint8)
    return Image.fromarray(spec_data)

  def image_to_audio(self, image: Image.Image) -> np.ndarray:
    spec_data = np.frombuffer(image.tobytes(), dtype=np.uint8).reshape((image.height, image.width))
    log_S = spec_data.astype("float") * self.top_db / 255 - self.top_db
    S = librosa.db_to_power(log_S)
    return librosa.feature.inverse.mel_to_audio(
      S, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length, n_iter=self.n_iter
    )