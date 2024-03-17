from typing import Iterable, Tuple

import numpy as np
import torch
from librosa.beat import beat_track
from PIL import Image
from tqdm.auto import tqdm

from audio_pipline import AudioDiffusionPipeline


class AudioDiffusion:
    def __init__(
        self,
        model_id: str = "NKumar5/CAGRock",
        cuda: bool = torch.cuda.is_available(),
        progress_bar: Iterable = tqdm,
        trust_remote_code: bool = False
    ):
      
        self.model_id = model_id
        self.pipe = AudioDiffusionPipeline.from_pretrained(self.model_id, trust_remote_code=trust_remote_code)
        if cuda:
            self.pipe.to("cuda")
        self.progress_bar = progress_bar or (lambda _: _)

    def generate_spectrogram_and_audio(
        self,
        steps: int = None,
        generator: torch.Generator = None,
        step_generator: torch.Generator = None,
        eta: float = 0,
        noise: torch.Tensor = None,
        encoding: torch.Tensor = None,
    ) -> Tuple[Image.Image, Tuple[int, np.ndarray]]:
        images, (sample_rate, audios) = self.pipe(
            batch_size=1,
            steps=steps,
            generator=generator,
            step_generator=step_generator,
            eta=eta,
            noise=noise,
            encoding=encoding,
            return_dict=False,
        )
        return images[0], (sample_rate, audios[0])

    def generate_spectrogram_and_audio_from_audio(
        self,
        audio_file: str = None,
        raw_audio: np.ndarray = None,
        slice: int = 0,
        start_step: int = 0,
        steps: int = None,
        generator: torch.Generator = None,
        mask_start_secs: float = 0,
        mask_end_secs: float = 0,
        step_generator: torch.Generator = None,
        eta: float = 0,
        encoding: torch.Tensor = None,
        noise: torch.Tensor = None,
    ) -> Tuple[Image.Image, Tuple[int, np.ndarray]]:
        images, (sample_rate, audios) = self.pipe(
            batch_size=1,
            audio_file=audio_file,
            raw_audio=raw_audio,
            slice=slice,
            start_step=start_step,
            steps=steps,
            generator=generator,
            mask_start_secs=mask_start_secs,
            mask_end_secs=mask_end_secs,
            step_generator=step_generator,
            eta=eta,
            noise=noise,
            encoding=encoding,
            return_dict=False,
        )
        return images[0], (sample_rate, audios[0])

    @staticmethod
    def loop_it(audio: np.ndarray, sample_rate: int, loops: int = 12) -> np.ndarray:
        _, beats = beat_track(y=audio, sr=sample_rate, units="samples")
        beats_in_bar = (len(beats) - 1) // 4 * 4
        if beats_in_bar > 0:
            return np.tile(audio[beats[0] : beats[beats_in_bar]], loops)
        return None
