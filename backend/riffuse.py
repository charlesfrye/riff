"""
Riffusion inference pipeline.

This is a modified version of the riffusion pipeline from
https://github.com/riffusion/riffusion

Here is the (MIT) license for the original code:

Copyright 2022 Hayk Martiros and Seth Forsgren

Permission is hereby granted, free of charge, to any person obtaining a copy of this
software and associated documentation files (the "Software"), to deal in the Software
without restriction, including without limitation the rights to use, copy, modify,
merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to the
following conditions:

The above copyright notice and this permission notice shall be included in all copies
or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
OR OTHER DEALINGS IN THE SOFTWARE.
"""
import dataclasses
from dataclasses import dataclass
from enum import Enum
import functools
import inspect
import io
import typing as T

import numpy as np
from scipy.io import wavfile
import torch
import torchaudio
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
from diffusers.schedulers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
import pydub

import utils


class RiffusionPipeline(DiffusionPipeline):
    """
    Diffusers pipeline for text-controlled img2img interpolation for audio generation.

    Part of this code was adapted from the non-img2img interpolation pipeline at:

        https://github.com/huggingface/diffusers/blob/main/examples/community/interpolate_stable_diffusion.py

    Check the documentation for DiffusionPipeline for full information.
    """

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: T.Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler],
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPFeatureExtractor,
    ):
        super().__init__()
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint: str,
        use_traced_unet: bool = True,
        channels_last: bool = False,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
        local_files_only: bool = False,
        low_cpu_mem_usage: bool = False,
        cache_dir: T.Optional[str] = None,
    ):
        """
        Load the riffusion model pipeline.

        Args:
            checkpoint: Model checkpoint on disk in diffusers format
            use_traced_unet: Whether to use the traced unet for speedups
            device: Device to load the model on
            channels_last: Whether to use channels_last memory format
            local_files_only: Don't download, only use local files
            low_cpu_mem_usage: Attempt to use less memory on CPU
        """
        pipeline = RiffusionPipeline.from_pretrained(
            checkpoint,
            revision="main",
            torch_dtype=dtype,
            # Disable the NSFW filter, causes incorrect false positives
            safety_checker=lambda images, **kwargs: (images, False),
            low_cpu_mem_usage=low_cpu_mem_usage,
            local_files_only=local_files_only,
            cache_dir=cache_dir,
        ).to(device)

        if channels_last:
            pipeline.unet.to(memory_format=torch.channels_last)

        # Optionally load a traced unet
        if checkpoint == "riffusion/riffusion-model-v1" and use_traced_unet:
            traced_unet = cls.load_traced_unet(
                checkpoint=checkpoint,
                subfolder="unet_traced",
                filename="unet_traced.pt",
                in_channels=pipeline.unet.in_channels,
                dtype=dtype,
                device=device,
                local_files_only=local_files_only,
                cache_dir=cache_dir,
            )

            if traced_unet is not None:
                pipeline.unet = traced_unet

        model = pipeline.to(device)

        return model

    @staticmethod
    def load_traced_unet(
        checkpoint: str,
        subfolder: str,
        filename: str,
        in_channels: int,
        dtype: torch.dtype,
        device: str = "cuda",
        local_files_only=False,
        cache_dir: T.Optional[str] = None,
    ) -> T.Optional[torch.nn.Module]:
        """
        Load a traced unet from the huggingface hub. This can improve performance.
        """
        # Download and load the traced unet
        unet_file = hf_hub_download(
            checkpoint,
            subfolder=subfolder,
            filename=filename,
            local_files_only=local_files_only,
            cache_dir=cache_dir,
        )
        unet_traced = torch.jit.load(unet_file)

        # Wrap it in a torch module
        class TracedUNet(torch.nn.Module):
            @dataclasses.dataclass
            class UNet2DConditionOutput:
                sample: torch.FloatTensor

            def __init__(self):
                super().__init__()
                self.in_channels = device
                self.device = device
                self.dtype = dtype

            def forward(self, latent_model_input, t, encoder_hidden_states):
                sample = unet_traced(latent_model_input, t, encoder_hidden_states)[0]
                return self.UNet2DConditionOutput(sample=sample)

        return TracedUNet()

    @property
    def device(self) -> str:
        return str(self.vae.device)

    @functools.lru_cache()
    def embed_text(self, text) -> torch.FloatTensor:
        """
        Takes in text and turns it into text embeddings.
        """
        text_input = self.tokenizer(
            text,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            embed = self.text_encoder(text_input.input_ids.to(self.device))[0]
        return embed

    @torch.no_grad()
    def riffuse(
        self,
        inputs: utils.InferenceInput,
    ) -> Image.Image:
        """
        Runs inference using interpolation with both img2img and text conditioning.

        Args:
            inputs: Parameter dataclass
            use_reweighting: Use prompt reweighting
        """
        generator = torch.Generator(device=self.device).manual_seed(inputs.seed)

        # Text encodings
        text_embedding = self.embed_text(inputs.prompt)

        # Image latents
        init_image_torch = preprocess_image(inputs.initImage).to(
            device=self.device, dtype=text_embedding.dtype
        )
        init_latent_dist = self.vae.encode(init_image_torch).latent_dist
        generator_latents = torch.Generator(device=self.device).manual_seed(0)

        init_latents = init_latent_dist.sample(generator=generator_latents)
        init_latents = 0.18215 * init_latents

        outputs = self.interpolate_img2img(
            text_embeddings=text_embedding,
            init_latents=init_latents,
            generator=generator,
            strength=inputs.denoising,
            num_inference_steps=inputs.numInferenceSteps,
            guidance_scale=inputs.guidance,
            negative_prompt=inputs.negativePrompt,
        )

        return outputs["images"][0]

    @torch.no_grad()
    def interpolate_img2img(
        self,
        text_embeddings: torch.Tensor,
        init_latents: torch.Tensor,
        generator: torch.Generator,
        strength: float = 0.8,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: T.Optional[T.Union[str, T.List[str]]] = None,
        num_images_per_prompt: int = 1,
        eta: T.Optional[float] = 0.0,
        **kwargs,
    ):
        batch_size = text_embeddings.shape[0]

        # set timesteps
        self.scheduler.set_timesteps(num_inference_steps)

        # duplicate text embeddings for each generation per prompt
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_images_per_prompt, 1)
        text_embeddings = text_embeddings.view(
            bs_embed * num_images_per_prompt, seq_len, -1
        )

        # `guidance_scale = 1` corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0
        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            if negative_prompt is None:
                uncond_tokens = [""]
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    "The length of `negative_prompt` should be equal to batch_size."
                )
            else:
                uncond_tokens = negative_prompt

            # max_length = text_input_ids.shape[-1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            uncond_embeddings = self.text_encoder(
                uncond_input.input_ids.to(self.device)
            )[0]

            # duplicate unconditional embeddings for each generation per prompt
            uncond_embeddings = uncond_embeddings.repeat_interleave(
                batch_size * num_images_per_prompt, dim=0
            )

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into
            # a single batch to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        latents_dtype = text_embeddings.dtype

        # get the original timestep using init_timestep
        offset = self.scheduler.config.get("steps_offset", 0)
        init_timestep = int(num_inference_steps * strength) + offset
        init_timestep = min(init_timestep, num_inference_steps)

        timesteps = self.scheduler.timesteps[-init_timestep]
        timesteps = torch.tensor(
            [timesteps] * batch_size * num_images_per_prompt, device=self.device
        )

        # add noise to latents using the timesteps
        noise = torch.randn(
            init_latents.shape,
            generator=generator,
            device=self.device,
            dtype=latents_dtype,
        )
        init_latents = self.scheduler.add_noise(init_latents, noise, timesteps)

        # prepare extra kwargs for the scheduler step, since not all schedulers
        # have the same args eta (η) is only used with the DDIMScheduler,
        # it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be in [0, 1]
        accepts_eta = "eta" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        latents = init_latents.clone()

        t_start = max(num_inference_steps - init_timestep + offset, 0)

        # Some schedulers like PNDM have timesteps as arrays
        # It's more optimized to move all timesteps to correct device beforehand
        timesteps = self.scheduler.timesteps[t_start:].to(self.device)

        for i, t in enumerate(self.progress_bar(timesteps)):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = (
                torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            )
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            noise_pred = self.unet(
                latent_model_input, t, encoder_hidden_states=text_embeddings
            ).sample

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(
                noise_pred, t, latents, **extra_step_kwargs
            ).prev_sample

        latents = 1.0 / 0.18215 * latents
        image = self.vae.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()

        image = self.numpy_to_pil(image)

        return dict(images=image, latents=latents, nsfw_content_detected=False)


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """
    Preprocess an image for the model.
    """
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=Image.LANCZOS)

    image_np = np.array(image).astype(np.float32) / 255.0
    image_np = image_np[None].transpose(0, 3, 1, 2)

    image_torch = torch.from_numpy(image_np)

    return 2.0 * image_torch - 1.0


@dataclass(frozen=True)
class SpectrogramParams:
    """
    Parameters for the conversion from audio to spectrograms to images and back.

    Includes helpers to convert to and from EXIF tags, allowing these parameters to
    be stored within spectrogram images.
    """

    # Whether the audio is stereo or mono
    stereo: bool = False

    # FFT parameters
    sample_rate: int = 44100
    step_size_ms: int = 10
    window_duration_ms: int = 100
    padded_duration_ms: int = 400

    # Mel scale parameters
    num_frequencies: int = 512
    min_frequency: int = 0
    max_frequency: int = 10000
    mel_scale_norm: T.Optional[str] = None
    mel_scale_type: str = "htk"
    max_mel_iters: int = 200

    # Griffin Lim parameters
    num_griffin_lim_iters: int = 32

    # Image parameterization
    power_for_image: float = 0.25

    class ExifTags(Enum):
        """
        Custom EXIF tags for the spectrogram image.
        """

        SAMPLE_RATE = 11000
        STEREO = 11005
        STEP_SIZE_MS = 11010
        WINDOW_DURATION_MS = 11020
        PADDED_DURATION_MS = 11030

        NUM_FREQUENCIES = 11040
        MIN_FREQUENCY = 11050
        MAX_FREQUENCY = 11060

        POWER_FOR_IMAGE = 11070
        MAX_VALUE = 11080

    @property
    def n_fft(self) -> int:
        """
        The number of samples in each STFT window, with padding.
        """
        return int(self.padded_duration_ms / 1000.0 * self.sample_rate)

    @property
    def win_length(self) -> int:
        """
        The number of samples in each STFT window.
        """
        return int(self.window_duration_ms / 1000.0 * self.sample_rate)

    @property
    def hop_length(self) -> int:
        """
        The number of samples between each STFT window.
        """
        return int(self.step_size_ms / 1000.0 * self.sample_rate)

    def to_exif(self) -> T.Dict[int, T.Any]:
        """
        Return a dictionary of EXIF tags for the current values.
        """
        return {
            self.ExifTags.SAMPLE_RATE.value: self.sample_rate,
            self.ExifTags.STEREO.value: self.stereo,
            self.ExifTags.STEP_SIZE_MS.value: self.step_size_ms,
            self.ExifTags.WINDOW_DURATION_MS.value: self.window_duration_ms,
            self.ExifTags.PADDED_DURATION_MS.value: self.padded_duration_ms,
            self.ExifTags.NUM_FREQUENCIES.value: self.num_frequencies,
            self.ExifTags.MIN_FREQUENCY.value: self.min_frequency,
            self.ExifTags.MAX_FREQUENCY.value: self.max_frequency,
            self.ExifTags.POWER_FOR_IMAGE.value: float(self.power_for_image),
        }

    @classmethod
    def from_exif(cls, exif: T.Mapping[int, T.Any]):
        """
        Create a SpectrogramParams object from the EXIF tags of the given image.
        """
        return cls(
            sample_rate=exif[cls.ExifTags.SAMPLE_RATE.value],
            stereo=bool(exif[cls.ExifTags.STEREO.value]),
            step_size_ms=exif[cls.ExifTags.STEP_SIZE_MS.value],
            window_duration_ms=exif[cls.ExifTags.WINDOW_DURATION_MS.value],
            padded_duration_ms=exif[cls.ExifTags.PADDED_DURATION_MS.value],
            num_frequencies=exif[cls.ExifTags.NUM_FREQUENCIES.value],
            min_frequency=exif[cls.ExifTags.MIN_FREQUENCY.value],
            max_frequency=exif[cls.ExifTags.MAX_FREQUENCY.value],
            power_for_image=exif[cls.ExifTags.POWER_FOR_IMAGE.value],
        )


class SpectrogramImageConverter:
    """
    Convert between spectrogram images and audio segments.

    This is a wrapper around SpectrogramConverter that additionally converts
    from spectrograms to images and back.

    The real audio processing lives in SpectrogramConverter.
    """

    def __init__(self, params: SpectrogramParams, device: str = "cuda"):
        self.p = params
        self.device = device
        self.converter = SpectrogramConverter(params=params, device=device)

    def spectrogram_image_from_audio(
        self,
        segment: pydub.AudioSegment,
    ) -> Image.Image:
        """
        Compute a spectrogram image from an audio segment.

        Args:
            segment: Audio segment to convert

        Returns:
            Spectrogram image (in pillow format)
        """
        assert int(segment.frame_rate) == self.p.sample_rate, "Sample rate mismatch"

        if self.p.stereo:
            if segment.channels == 1:
                print("WARNING: Mono audio but stereo=True, cloning channel")
                segment = segment.set_channels(2)
            elif segment.channels > 2:
                print("WARNING: Multi channel audio, reducing to stereo")
                segment = segment.set_channels(2)
        else:
            if segment.channels > 1:
                print("WARNING: Stereo audio but stereo=False, setting to mono")
                segment = segment.set_channels(1)

        spectrogram = self.converter.spectrogram_from_audio(segment)

        image = image_from_spectrogram(
            spectrogram,
            power=self.p.power_for_image,
        )

        # Store conversion params in exif metadata of the image
        exif_data = self.p.to_exif()
        exif_data[SpectrogramParams.ExifTags.MAX_VALUE.value] = float(
            np.max(spectrogram)
        )
        exif = image.getexif()
        exif.update(exif_data.items())

        return image

    def audio_from_spectrogram_image(
        self,
        image: Image.Image,
        apply_filters: bool = True,
        max_value: float = 30e6,
    ) -> pydub.AudioSegment:
        """
        Reconstruct an audio segment from a spectrogram image.

        Args:
            image: Spectrogram image (in pillow format)
            apply_filters: Apply post-processing to improve the reconstructed audio
            max_value: Scaled max amplitude of the spectrogram. Shouldn't matter.
        """
        spectrogram = spectrogram_from_image(
            image,
            max_value=max_value,
            power=self.p.power_for_image,
            stereo=self.p.stereo,
        )

        segment = self.converter.audio_from_spectrogram(
            spectrogram,
            apply_filters=apply_filters,
        )

        return segment


class SpectrogramConverter:
    """
    Convert between audio segments and spectrogram tensors using torchaudio.

    In this class a "spectrogram" is defined as a (batch, time, frequency) tensor with
    float values that represent the amplitude of the frequency at that time bucket
    (in the frequency domain). Frequencies are given in the perceptul Mel scale defined
    by the params. A more specific term used in some functions is "mel amplitudes".

    The spectrogram computed from `spectrogram_from_audio` is complex valued, but it
    only     returns the amplitude, because the phase is chaotic and hard to learn.
    The function `audio_from_spectrogram` is an approximate inverse of
    `spectrogram_from_audio`, which approximates the phase information using the
    Griffin-Lim algorithm.

    Each channel in the audio is treated independently, and the spectrogram has a
    batch dimension equal to the number of channels in the input audio segment.

    Both the Griffin Lim algorithm and the Mel scaling process are lossy.

    For more information, see https://pytorch.org/audio/stable/transforms.html
    """

    def __init__(self, params: SpectrogramParams, device: str = "cuda"):
        self.p = params
        self.device = device

        # https://pytorch.org/audio/stable/generated/torchaudio.transforms.Spectrogram.html
        self.spectrogram_func = torchaudio.transforms.Spectrogram(
            n_fft=params.n_fft,
            hop_length=params.hop_length,
            win_length=params.win_length,
            pad=0,
            window_fn=torch.hann_window,
            power=None,
            normalized=False,
            wkwargs=None,
            center=True,
            pad_mode="reflect",
            onesided=True,
        ).to(self.device)

        # https://pytorch.org/audio/stable/generated/torchaudio.transforms.GriffinLim.html
        self.inverse_spectrogram_func = torchaudio.transforms.GriffinLim(
            n_fft=params.n_fft,
            n_iter=params.num_griffin_lim_iters,
            win_length=params.win_length,
            hop_length=params.hop_length,
            window_fn=torch.hann_window,
            power=1.0,
            wkwargs=None,
            momentum=0.99,
            length=None,
            rand_init=True,
        ).to(self.device)

        # https://pytorch.org/audio/stable/generated/torchaudio.transforms.MelScale.html
        self.mel_scaler = torchaudio.transforms.MelScale(
            n_mels=params.num_frequencies,
            sample_rate=params.sample_rate,
            f_min=params.min_frequency,
            f_max=params.max_frequency,
            n_stft=params.n_fft // 2 + 1,
            norm=params.mel_scale_norm,
            mel_scale=params.mel_scale_type,
        ).to(self.device)

        # https://pytorch.org/audio/stable/generated/torchaudio.transforms.InverseMelScale.html
        self.inverse_mel_scaler = torchaudio.transforms.InverseMelScale(
            n_stft=params.n_fft // 2 + 1,
            n_mels=params.num_frequencies,
            sample_rate=params.sample_rate,
            f_min=params.min_frequency,
            f_max=params.max_frequency,
            max_iter=params.max_mel_iters,
            tolerance_loss=1e-5,
            tolerance_change=1e-8,
            sgdargs=None,
            norm=params.mel_scale_norm,
            mel_scale=params.mel_scale_type,
        ).to(self.device)

    def spectrogram_from_audio(
        self,
        audio: pydub.AudioSegment,
    ) -> np.ndarray:
        """
        Compute a spectrogram from an audio segment.

        Args:
            audio: Audio segment which must match the sample rate of the params

        Returns:
            spectrogram: (channel, frequency, time)
        """
        assert (
            int(audio.frame_rate) == self.p.sample_rate
        ), "Audio sample rate must match params"

        # Get the samples as a numpy array in (batch, samples) shape
        waveform = np.array([c.get_array_of_samples() for c in audio.split_to_mono()])

        # Convert to floats if necessary
        if waveform.dtype != np.float32:
            waveform = waveform.astype(np.float32)

        waveform_tensor = torch.from_numpy(waveform).to(self.device)
        amplitudes_mel = self.mel_amplitudes_from_waveform(waveform_tensor)
        return amplitudes_mel.cpu().numpy()

    def audio_from_spectrogram(
        self,
        spectrogram: np.ndarray,
        apply_filters: bool = True,
    ) -> pydub.AudioSegment:
        """
        Reconstruct an audio segment from a spectrogram.

        Args:
            spectrogram: (batch, frequency, time)
            apply_filters: Post-process with normalization and compression

        Returns:
            audio: Audio segment with channels equal to the batch dimension
        """
        # Move to device
        amplitudes_mel = torch.from_numpy(spectrogram).to(self.device)

        # Reconstruct the waveform
        waveform = self.waveform_from_mel_amplitudes(amplitudes_mel)

        # Convert to audio segment
        segment = audio_from_waveform(
            samples=waveform.cpu().numpy(),
            sample_rate=self.p.sample_rate,
            # Normalize the waveform to the range [-1, 1]
            normalize=True,
        )

        # Optionally apply post-processing filters
        if apply_filters:
            segment = apply_audio_filters(
                segment,
                compression=False,
            )

        return segment

    def mel_amplitudes_from_waveform(
        self,
        waveform: torch.Tensor,
    ) -> torch.Tensor:
        """
        Torch-only function to compute Mel-scale amplitudes from a waveform.

        Args:
            waveform: (batch, samples)

        Returns:
            amplitudes_mel: (batch, frequency, time)
        """
        # Compute the complex-valued spectrogram
        spectrogram_complex = self.spectrogram_func(waveform)

        # Take the magnitude
        amplitudes = torch.abs(spectrogram_complex)

        # Convert to mel scale
        return self.mel_scaler(amplitudes)

    def waveform_from_mel_amplitudes(
        self,
        amplitudes_mel: torch.Tensor,
    ) -> torch.Tensor:
        """
        Torch-only function to approx reconstruct a waveform from Mel-scale amplitudes.

        Args:
            amplitudes_mel: (batch, frequency, time)

        Returns:
            waveform: (batch, samples)
        """
        # Convert from mel scale to linear
        amplitudes_linear = self.inverse_mel_scaler(amplitudes_mel)

        # Run the approximate algorithm to compute the phase and recover the waveform
        return self.inverse_spectrogram_func(amplitudes_linear)


def audio_from_waveform(
    samples: np.ndarray, sample_rate: int, normalize: bool = False
) -> pydub.AudioSegment:
    """
    Convert a numpy array of samples of a waveform to an audio segment.

    Args:
        samples: (channels, samples) array
    """
    # Normalize volume to fit in int16
    if normalize:
        samples *= np.iinfo(np.int16).max / np.max(np.abs(samples))

    # Transpose and convert to int16
    samples = samples.transpose(1, 0)
    samples = samples.astype(np.int16)

    # Write to the bytes of a WAV file
    wav_bytes = io.BytesIO()
    wavfile.write(wav_bytes, sample_rate, samples)
    wav_bytes.seek(0)

    # Read into pydub
    return pydub.AudioSegment.from_wav(wav_bytes)


def apply_audio_filters(
    segment: pydub.AudioSegment, compression: bool = False
) -> pydub.AudioSegment:
    """
    Apply post-processing filters to the audio segment to compress it and
    keep at a -10 dBFS level.
    """

    if compression:
        segment = pydub.effects.normalize(
            segment,
            headroom=0.1,
        )

        segment = segment.apply_gain(-10 - segment.dBFS)

        segment = pydub.effects.compress_dynamic_range(
            segment,
            threshold=-20.0,
            ratio=4.0,
            attack=5.0,
            release=50.0,
        )

    desired_db = -12
    segment = segment.apply_gain(desired_db - segment.dBFS)

    segment = pydub.effects.normalize(
        segment,
        headroom=0.1,
    )

    return segment


def image_from_spectrogram(spectrogram: np.ndarray, power: float = 0.25) -> Image.Image:
    """
    Compute a spectrogram image from a spectrogram magnitude array.

    This is the inverse of spectrogram_from_image, except for discretization error from
    quantizing to uint8.

    Args:
        spectrogram: (channels, frequency, time)
        power: A power curve to apply to the spectrogram to preserve contrast

    Returns:
        image: (frequency, time, channels)
    """
    # Rescale to 0-1
    max_value = np.max(spectrogram)
    data = spectrogram / max_value

    # Apply the power curve
    data = np.power(data, power)

    # Rescale to 0-255
    data = data * 255

    # Invert
    data = 255 - data

    # Convert to uint8
    data = data.astype(np.uint8)

    # Munge channels into a PIL image
    if data.shape[0] == 1:
        image = Image.fromarray(data[0], mode="L").convert("RGB")
    elif data.shape[0] == 2:
        data = np.array([np.zeros_like(data[0]), data[0], data[1]]).transpose(1, 2, 0)
        image = Image.fromarray(data, mode="RGB")
    else:
        raise NotImplementedError(f"Unsupported number of channels: {data.shape[0]}")

    # Flip Y
    image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

    return image


def spectrogram_from_image(
    image: Image.Image,
    power: float = 0.25,
    stereo: bool = False,
    max_value: float = 30e6,
) -> np.ndarray:
    """
    Compute a spectrogram magnitude array from a spectrogram image.

    This is the inverse of image_from_spectrogram, except for discretization error from
    quantizing to uint8.

    Args:
        image: (frequency, time, channels)
        power: The power curve applied to the spectrogram
        stereo: Whether the spectrogram encodes stereo data
        max_value: The max value of the original spectrogram.

    Returns:
        spectrogram: (channels, frequency, time)
    """
    # Convert to RGB if single channel
    if image.mode in ("P", "L"):
        image = image.convert("RGB")

    # Flip Y
    image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

    # Munge channels into a numpy array of (channels, frequency, time)
    data = np.array(image).transpose(2, 0, 1)
    if stereo:
        # Take the G and B channels as done in image_from_spectrogram
        data = data[[1, 2], :, :]
    else:
        data = data[0:1, :, :]

    # Convert to floats
    data = data.astype(np.float32)

    # Invert
    data = 255 - data

    # Rescale to 0-1
    data = data / 255

    # Reverse the power curve
    data = np.power(data, 1 / power)

    # Rescale to max value
    data = data * max_value

    return data
