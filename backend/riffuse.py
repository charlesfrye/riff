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
import functools
import inspect
import typing as T

import numpy as np
import torch
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
from diffusers.schedulers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

import utils


# TODO: cleanup
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

    @functools.lru_cache()
    def embed_text_weighted(self, text) -> torch.FloatTensor:
        """
        Get text embedding with weights.
        """
        # TODO: get this function from their code
        return get_weighted_text_embeddings(  # noqa: F821
            pipe=self,
            prompt=text,
            uncond_prompt=None,
            max_embeddings_multiples=3,
            no_boseos_middle=False,
            skip_parsing=False,
            skip_weighting=False,
        )[0]

    @torch.no_grad()
    def riffuse(
        self,
        inputs: utils.InferenceInput,
        use_reweighting: bool = True,
    ) -> Image.Image:
        """
        Runs inference using interpolation with both img2img and text conditioning.

        Args:
            inputs: Parameter dataclass
            use_reweighting: Use prompt reweighting
        """
        generator = torch.Generator(device=self.device).manual_seed(inputs.seed)

        # Text encodings
        if use_reweighting:
            embed_start = self.embed_text_weighted(inputs.prompt)
        else:
            embed_start = self.embed_text(inputs.prompt)

        text_embedding = embed_start

        # Image latents
        init_image_torch = preprocess_image(inputs.init_image).to(
            device=self.device, dtype=embed_start.dtype
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
            num_inference_steps=inputs.num_inference_steps,
            guidance_scale=inputs.guidance,
            negative_prompt=inputs.negative_prompt,
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
