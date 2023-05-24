from typing import Optional
from dataclasses import dataclass

from PIL.Image import Image
import pydub


@dataclass(frozen=True)
class InferenceInput:
    """
    Parameters for a single run of the riffusion model.
    """

    initImage: Image
    prompt: str = "Recorder version of Star Wars theme song"
    seed: int = 42
    negativePrompt: Optional[str] = None
    denoising: float = 0.75
    guidance: float = 7.0
    numInferenceSteps: int = 50


@dataclass(frozen=True)
class InferenceRequest:
    """
    This is the API required for a request to the model server.
    """

    initAudio: pydub.AudioSegment
    prompt: str = "Recorder version of Star Wars theme song"
    seed: int = 42
    negativePrompt: Optional[str] = None
    denoising: float = 0.75
    guidance: float = 7.0
    numInferenceSteps: int = 50
