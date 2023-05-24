/* eslint-disable no-console */
const inferenceInput = {};
const backendUrl = "https://charlesfrye--riff.modal.run";

document.addEventListener("DOMContentLoaded", () => {
  const button = document.querySelector("button");
  const body = document.querySelector("body");
  const audioFileInput = document.querySelector("#audio-file-input");

  async function runInference() {
    // TODO: check that it's valid before calling the backend with fetch
    const response = await fetch(backendUrl, {
      method: "POST",
      mode: "cors",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(inferenceInput),
    });

    const responseJson = await response.json();
    // eslint-disable-next-line no-unused-vars
    const { audioUrl, imageUrl } = responseJson;

    // add audio with controls
    const audioElement = document.createElement("audio");
    audioElement.src = audioUrl;
    audioElement.type = "audio/mpeg";
    audioElement.controls = true;

    body.appendChild(audioElement);
  }

  button.addEventListener("click", runInference);

  audioFileInput.addEventListener("change", (event) => {
    inferenceInput.setInitAudio(event.target.files[0]);
  });
});

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

// User input for initial audio
inferenceInput.setInitAudio = (file) => {
  if (file instanceof File && file.type.startsWith("audio/")) {
    const reader = new FileReader();
    reader.onloadend = () => {
      const base64data = reader.result;
      inferenceInput.initAudio = base64data;
      console.log(inferenceInput);
    };
    reader.readAsDataURL(file);
  } else {
    console.error("Input must be a File.");
  }
};

// eslint-disable-next-line no-unused-vars
async function setPlaceholders() {
  inferenceInput.setPrompt("Star Wars recorder music");
  inferenceInput.setSeed(27);
  inferenceInput.setNegativePrompt("Lord of the Rings Organ Music");
  inferenceInput.setDenoise(42);
  inferenceInput.setGuidance(7.3);
  inferenceInput.setNumInferenceSteps(9);
  // Open file dialog
  const fileInput = document.querySelector("#audio-file-input");
  fileInput.dispatchEvent(new MouseEvent("click"));

  // Listen for the file input change event
  fileInput.addEventListener("change", (event) => {
    // When a file is selected, set it as the initial audio
    inferenceInput.setInitAudio(event.target.files[0]);
    console.log(inferenceInput);
  });
}
