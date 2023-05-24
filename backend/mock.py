import requests

backend_url = "https://charlesfrye--riff.modal.run"

with open("test.m4a.b64", "r") as f:
    init_audio = f.read()

# build a request that looks like an InferenceRequest
data = {
    "initAudio": init_audio,
    "prompt": "lofi beat to study and relax to",
    "seed": 117,
    "denoising": 0.75,
    "guidance": 7.0,
    "numInferenceSteps": 50,
}

headers = {"Content-Type": "application/json"}

response = requests.post(url=backend_url, json=data, headers=headers)

response.raise_for_status()

print(response.json())
