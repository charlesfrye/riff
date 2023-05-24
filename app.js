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
  const promptInput = document.querySelector("#prompt-input");
  if (typeof text === "string" && text.trim().length > 0) {
    inferenceInput.prompt = text;
    promptInput.classList.remove("error-input");
  } else {
    delete inferenceInput.prompt;
    promptInput.classList.add("error-input");
    console.log("Invalid input: Please enter a non-empty string value.");
  }
};

// User input for seed
inferenceInput.setSeed = (seed) => {
  const seedInput = document.querySelector("#seed-input");
  if (Number.isInteger(seed)) {
    inferenceInput.seed = seed;
    seedInput.classList.remove("error-input");
  } else {
    seedInput.classList.add("error-input");
  }
};

// User input for negative prompt (optional)
inferenceInput.setNegativePrompt = (text) => {
  const negativePromptInput = document.querySelector("#negative-prompt-input");
  if (typeof text === "string") {
    if (text.length > 0) {
      inferenceInput.negativePrompt = text;
      negativePromptInput.classList.remove("error-input");
    } else {
      delete inferenceInput.negativePrompt;
      negativePromptInput.classList.remove("error-input");
    }
  } else {
    negativePromptInput.classList.add("error-input");
  }
};

// User input for denoising strength
inferenceInput.setDenoising = (denoising) => {
  const denoisingInput = document.querySelector("#denoising-input");
  const parsedFloat = Number.parseFloat(denoising);
  if (!Number.isNaN(parsedFloat) && parsedFloat >= 0) {
    inferenceInput.denoising = parsedFloat;
    denoisingInput.classList.remove("error-input");
  } else {
    denoisingInput.classList.add("error-input");
  }
};

// User input for guidance strength
inferenceInput.setGuidance = (guidance) => {
  const guidanceInput = document.querySelector("#guidance-input");
  const parsedFloat = Number.parseFloat(guidance);
  if (!Number.isNaN(parsedFloat) && parsedFloat >= 0) {
    inferenceInput.guidance = parsedFloat;
    guidanceInput.classList.remove("error-input");
  } else {
    guidanceInput.classList.add("error-input");
  }
};

// User input for number of inference steps
inferenceInput.setNumInferenceSteps = (numInferenceSteps) => {
  const numInferenceStepsInput = document.querySelector(
    "#num-inference-steps-input"
  );
  if (Number.isInteger(numInferenceSteps) && numInferenceSteps >= 0) {
    inferenceInput.numInferenceSteps = numInferenceSteps;
    numInferenceStepsInput.classList.remove("error-input");
  } else {
    numInferenceStepsInput.classList.add("error-input");
  }
};

// User input for initial audio
inferenceInput.setInitAudio = (file) => {
  const audioFileInput = document.querySelector("#audio-file-input");
  if (file instanceof File && file.type.startsWith("audio/")) {
    const reader = new FileReader();
    reader.onloadend = () => {
      const base64data = reader.result;
      inferenceInput.initAudio = base64data;
      audioFileInput.classList.remove("error-input");
    };
    reader.readAsDataURL(file);
  } else {
    audioFileInput.classList.add("error-input");
  }
};

// eslint-disable-next-line no-unused-vars
async function setPlaceholders() {
  inferenceInput.setPrompt("Star Wars recorder music");
  inferenceInput.setSeed(27);
  inferenceInput.setNegativePrompt("Lord of the Rings Organ Music");
  inferenceInput.setDenoising(42);
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

function setDefaults() {
  // Set default values
  const defaultSeed = 117;
  const defaultDenoising = 0.8;
  const defaultGuidance = 7.0;
  const defaultNumInferenceSteps = 50;
  // Set inferenceInput to default values
  inferenceInput.setSeed(defaultSeed);
  inferenceInput.setDenoising(defaultDenoising);
  inferenceInput.setGuidance(defaultGuidance);
  inferenceInput.setNumInferenceSteps(defaultNumInferenceSteps);
  // Set input box values to default values
  document.querySelector("#seed-input").value = defaultSeed;
  document.querySelector("#denoising-input").value = defaultDenoising;
  document.querySelector("#guidance-input").value = defaultGuidance;
  document.querySelector("#num-inference-steps-input").value =
    defaultNumInferenceSteps;
}

// Adding buttons
document.addEventListener("DOMContentLoaded", () => {
  document.querySelector("#set-prompt-button").addEventListener("click", () => {
    const prompt = document.querySelector("#prompt-input").value;
    console.log(`Setting prompt to: ${prompt}`);
    inferenceInput.setPrompt(prompt);
    console.log(`New inferenceInput: ${JSON.stringify(inferenceInput)}`);
  });

  document.querySelector("#set-seed-button").addEventListener("click", () => {
    const seed = Number(document.querySelector("#seed-input").value);
    inferenceInput.setSeed(seed);
  });

  document
    .querySelector("#set-negative-prompt-button")
    .addEventListener("click", () => {
      const negativePrompt = document.querySelector(
        "#negative-prompt-input"
      ).value;
      inferenceInput.setNegativePrompt(negativePrompt);
    });

  document
    .querySelector("#set-denoising-button")
    .addEventListener("click", () => {
      const denoising = parseFloat(
        document.querySelector("#denoising-input").value
      );
      inferenceInput.setDenoising(denoising);
    });

  document
    .querySelector("#set-guidance-button")
    .addEventListener("click", () => {
      const guidance = parseFloat(
        document.querySelector("#guidance-input").value
      );
      inferenceInput.setGuidance(guidance);
    });

  document
    .querySelector("#set-num-inference-steps-button")
    .addEventListener("click", () => {
      const numInferenceSteps = Number(
        document.querySelector("#num-inference-steps-input").value
      );
      inferenceInput.setNumInferenceSteps(numInferenceSteps);
    });
  setDefaults();
});
