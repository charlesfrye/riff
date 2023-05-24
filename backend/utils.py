from typing import Optional
from dataclasses import dataclass

from PIL.Image import Image
import pydub


@dataclass(frozen=True)
class InferenceInput:
    """
    Parameters for a single run of the riffusion model.
    """

    init_image: Image
    prompt: str = "Recorder version of Star Wars theme song"
    seed: int = 42
    negative_prompt: Optional[str] = None
    denoising: float = 0.75
    guidance: float = 7.0
    num_inference_steps: int = 50


@dataclass(frozen=True)
class InferenceRequest:
    """
    This is the API required for a request to the model server.
    """

    init_audio: pydub.AudioSegment
    prompt: str = "Recorder version of Star Wars theme song"
    seed: int = 42
    negative_prompt: Optional[str] = None
    denoising: float = 0.75
    guidance: float = 7.0
    num_inference_steps: int = 50
