"""Modal app for prompt-guided audio-to-audio transformation."""
from pathlib import Path

import modal

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(  # python dependencies for riffusion inference
        "accelerate",
        "boto3",
        "diffusers>=0.9.0",
        "numpy",
        "pillow>=9.1.0",
        "pydub",
        "pysoundfile",
        "scipy",
        "soundfile",
        "smart_open[aws]",
        "torch",
        "torchaudio",
        "torchvision",
        "transformers",
    )
    .apt_install("ffmpeg")  # system lib dependencies for riffusion inference
)

volume = modal.SharedVolume().persist("riffusion-models")
MODEL_DIR = Path("/model")

stub = modal.Stub(
    name="riffusion",
    secrets=[
        # this is where we add API keys, passwords, and URLs, which are stored on Modal
        modal.Secret.from_name("aws-personal")
    ],
    mounts=[
        # we make our local modules available to the containers
        *modal.create_package_mounts(module_names=["riffuse", "utils"]),
    ],
)


@stub.function(image=image, gpu="A10G", shared_volumes={str(MODEL_DIR): volume})
@modal.web_endpoint(method="POST", label="riff")
def inference(request_config: dict):
    from utils import InferenceInput

    # parse out inference configuration from request
    init_audio = base64_to_audio(request_config.pop("initAudio"))
    init_image = audio_to_image(init_audio)
    inference_config = InferenceInput(initImage=init_image, **request_config)

    # call out to the GPU-powered inference function on Modal
    image = Riffuser().inference.call(inference_config)
    if image is None:
        image = init_image

    # convert the spectrogram image into audio
    audio = image_to_audio(image)

    audio_bytes = audio_to_bytes(audio)
    image_bytes = image_to_bytes(image)

    # send the audio and the image to S3
    audio_url = send_to_s3(audio_bytes, filename="riff.mp3")
    image_url = send_to_s3(image_bytes, filename="spectrogram.jpg")

    # return the URLs of the audio and the image

    return {
        "audioUrl": audio_url,
        "imageUrl": image_url,
    }


@stub.cls(
    image=image,
    gpu="A10G",
    shared_volumes={str(MODEL_DIR): volume},
)
class Riffuser:
    def __enter__(self):
        from riffuse import RiffusionPipeline

        pipe = RiffusionPipeline.load_checkpoint(
            checkpoint="riffusion/riffusion-model-v1",
            use_traced_unet=True,
            device="cuda",
            cache_dir=MODEL_DIR,
        )

        self.pipe = pipe

    @modal.method()
    def inference(self, inputs):
        image = self.pipe.riffuse(inputs)

        return image


@stub.function(image=image, gpu="A10G")
def image_to_audio(image):
    """Convert a spectrogram image into audio"""
    import riffuse

    params = riffuse.SpectrogramParams(
        min_frequency=0,
        max_frequency=10000,
    )

    converter = riffuse.SpectrogramImageConverter(params=params, device="cuda")
    audio = converter.audio_from_spectrogram_image(
        image,
        apply_filters=True,
    )

    return audio


@stub.function(image=image, gpu="A10G")
def audio_to_image(audio):
    """Convert audio into a spectrogram image"""
    import riffuse

    params = riffuse.SpectrogramParams(
        min_frequency=0,
        max_frequency=10000,
    )

    converter = riffuse.SpectrogramImageConverter(params=params, device="cuda")
    image = converter.spectrogram_image_from_audio(audio)

    return image


@stub.function(image=image)
def audio_to_bytes(audio):
    """Export pydub audio to MP3 bytes."""
    import io

    mp3_bytes = io.BytesIO()
    audio.export(mp3_bytes, format="mp3")
    mp3_bytes.seek(0)

    return mp3_bytes


@stub.function(image=image)
def base64_to_audio(string):
    """Decode base64 string into pydub audio."""
    import base64
    import io

    from pydub import AudioSegment

    # handle data URLs
    if string.startswith("data:audio/"):
        # remove the data prefix but keep the filetype, in case we need it later
        string = string[string.index("/") :]
        # remove the rest of the prefix from the string
        string = string[string.index(",") + 1 :]

    audio_bytes = io.BytesIO(base64.b64decode(string))
    return AudioSegment.from_file(audio_bytes)


@stub.function(image=image)
def image_to_bytes(image):
    """Export image to JPEG bytes."""
    import io

    image_bytes = io.BytesIO()
    image.save(image_bytes, exif=image.getexif(), format="JPEG")
    image_bytes.seek(0)

    return image_bytes


@stub.function(image=image)
def send_to_s3(bytes, filename=None):
    """Upload bytes to S3 and return the URL."""
    from uuid import uuid4

    import boto3
    from smart_open import open

    # set up AWS auth with boto3
    session = boto3.Session()
    transport_params = {"client": session.client("s3")}

    # use uuid to set filenames
    idstr = str(uuid4())[:8]
    filename = f"-{idstr}.".join(filename.split("."))

    with open(
        f"s3://riff-store/{filename}", "wb", transport_params=transport_params
    ) as f:
        f.write(bytes.read())

    return f"https://riff-store.s3.us-west-2.amazonaws.com/{filename}"
