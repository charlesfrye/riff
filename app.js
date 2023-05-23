const backendUrl = "https://charlesfrye--riff.modal.run";

document.addEventListener("DOMContentLoaded", () => {
  const button = document.querySelector("button");
  const body = document.querySelector("body");

  async function runInference() {
    // TODO: set this up to take an InferenceInput object
    //       and check that it's valid before calling the backend with fetch
    const response = await fetch(backendUrl, {
      method: "POST",
      mode: "cors",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        // TODO: make this an InferenceInput object
        // Text prompt fed into a CLIP model
        prompt: "Recorder version of Star Wars theme song", // str

        // Random seed for noise
        seed: 42, // int

        // Negative prompt to avoid (optional)
        negative_prompt: "sounds good", // str or not defined

        // Denoising strength
        denoising: 0.75, // float

        // Strength of the prompt
        guidance: 7.0, // float

        // Number of steps to run (trades quality and speed)
        num_inference_steps: 50, // int

        // Seed image # TODO what does this one look like?
        seed_image: undefined, // ?
      }),
    });

    const responseJson = await response.json();
    const audioUrl = responseJson.url;

    // add audio with controls
    const audioElement = document.createElement("audio");
    audioElement.src = audioUrl;
    audioElement.type = "audio/mpeg";
    audioElement.controls = true;

    body.appendChild(audioElement);
  }

  button.addEventListener("click", runInference);
});
