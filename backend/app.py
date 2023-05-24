"""Modal app for prompt-guided audio-to-audio transformation."""
import modal

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(  # python dependencies for riffusion inference
        "accelerate",
        "diffusers>=0.9.0",
        "numpy",
        "pillow>=9.1.0",
        "pydub",
        "pysoundfile",
        "scipy",
        "soundfile",
        "torch",
        "torchaudio",
        "torchvision",
        "transformers",
    )
    .apt_install("ffmpeg")  # system lib dependencies for riffusion inference
)

stub = modal.Stub(
    name="riffusion",
    secrets=[
        # this is where we add API keys, passwords, and URLs, which are stored on Modal
    ],
    mounts=[
        # we make our local modules available to the containers
        *modal.create_package_mounts(module_names=["riffuse", "utils"])
    ],
)


@stub.function(image=image)
@modal.web_endpoint(method="POST", label="riff")
def inference(request_config: dict):
    testing = True

    from utils import InferenceInput

    # parse out inference configuration from request
    if not testing:
        init_audio = bytes_to_audio(request_config.pop("init_audio"))
        init_image = audio_to_image(init_audio)
        inference_config = InferenceInput(init_image=init_image, **request_config)

        # call out to the GPU-powered inference function on Modal
        image = Riffuser().inference.call(inference_config)

        # convert the spectrogram image into audio
        audio = image_to_audio(image)

        audio_bytes = audio_to_bytes(audio)
        image_bytes = image_to_bytes(image)
    else:
        audio_bytes, image_bytes = [], []

    # send the audio and the image to S3
    audio_url = send_to_s3(audio_bytes, filename="riff.mp3")
    image_url = send_to_s3(image_bytes, filename="spectrogram.jpg")

    # return the URLs of the audio and the image

    return {
        "url": "https://riff-store.s3.us-west-2.amazonaws.com/test.mp3",
        "audioUrl": audio_url,
        "imageUrl": image_url,
    }


# TODO: store model weights with a SharedVolume
@stub.cls(
    image=image,
    gpu="A10G",
)
class Riffuser:
    def __enter__(self):
        from riffuse import RiffusionPipeline

        pipe = RiffusionPipeline.load_checkpoint(
            checkpoint="riffusion/riffusion-model-v1",
            use_traced_unet=True,
            device="cuda",
        )

        self.pipe = pipe

    @modal.method()
    def inference(self, inputs):
        image = self.pipe.riffuse(inputs)

        return image


@stub.function(image=image)
def image_to_audio(image):
    """Convert a spectrogram image into audio"""
    # TODO: get spectrogram tools working
    params = SpectrogramParams(  # noqa: F821
        min_frequency=0,
        max_frequency=10000,
    )

    converter = SpectrogramImageConverter(params=params, device="cuda")  # noqa: F821
    audio = converter.audio_from_spectrogram_image(
        image,
        apply_filters=True,
    )

    return audio


@stub.function(image=image)
def audio_to_image(audio):
    """Convert audio into a spectrogram image"""
    # TODO: get spectrogram tools working
    image = None
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
def bytes_to_audio(bytes):
    """Decode base64 bytes into pydub audio."""
    # TODO: implement
    pass


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
    # TODO: implement
    # TODO: use hashing to set filenames
    if filename.endswith(".jpg"):
        return "https://riff-store.s3.us-west-2.amazonaws.com/test.jpg"
    return "https://riff-store.s3.us-west-2.amazonaws.com/test.mp3"
