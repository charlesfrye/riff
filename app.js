const inferenceInput = {};
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
        seed_audio: undefined, // ?
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

/* function to take user input and pass it to the backend
  - get user input
  - get user upload
  - get user seed
  - get user negative prompt (optional)
  - get user denoise
  - pass it to backend
  - backend returns audio
  - add audio to page

function to call runInference, hands off state of object to JSON
*/
// User input for prompt
inferenceInput.setPrompt = (text) => {
  if (typeof text === "string" && text.trim().length > 0) {
    inferenceInput.prompt = text;
    console.log(inferenceInput);
  } else {
    console.error("Prompt must be a non-empty string");
  }
};

// User input for seed
inferenceInput.setSeed = (seed) => {
  if (Number.isInteger(seed)) {
    inferenceInput.seed = seed;
    console.log(inferenceInput);
  } else {
    console.error("Input must be an integer.");
  }
};

// User input for negative prompt (optional)
inferenceInput.setNegativePrompt = (text) => {
  if (typeof text === "string") {
    if (text.length > 0) {
      inferenceInput.negativePrompt = text;
    } else {
      delete inferenceInput.negativePrompt;
      console.log("Deleted negative prompt");
    }
    console.log(inferenceInput);
  } else {
    console.error("Prompt must be a string");
  }
};

// User input for denoising strength
inferenceInput.setDenoise = (denoise) => {
  const parsedFloat = Number.parseFloat(denoise);
  if (!Number.isNaN(parsedFloat) && parsedFloat >= 0) {
    inferenceInput.denoise = parsedFloat;
    console.log(inferenceInput);
  } else {
    console.error("Input must be a positive float");
  }
};

// User input for guidance strength
inferenceInput.setGuidance = (guidance) => {
  const parsedFloat = Number.parseFloat(guidance);
  if (!Number.isNaN(parsedFloat) && parsedFloat >= 0) {
    inferenceInput.guidance = parsedFloat;
    console.log(inferenceInput);
  } else {
    console.error("Input must be a positive float");
  }
};

// User input for number of inference steps
inferenceInput.setNumInferenceSteps = (numInferenceSteps) => {
  if (Number.isInteger(numInferenceSteps) && numInferenceSteps >= 0) {
    inferenceInput.numInferenceSteps = numInferenceSteps;
    console.log(inferenceInput);
  } else {
    console.error("Input must be an integer.");
  }
};
