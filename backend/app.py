import modal

image = modal.Image.debian_slim(python_version="3.10")

stub = modal.Stub(name="riffusion")


@stub.function(image=image)
@modal.web_endpoint(method="POST", label="riff")
def inference(input: dict):
    return {"url": "https://riff-store.s3.us-west-2.amazonaws.com/test.mp3"}
