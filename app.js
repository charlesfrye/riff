/* eslint-disable no-unused-vars */
/* eslint-disable no-plusplus */
/* eslint-disable no-alert */
/* eslint-disable no-console */
const inferenceInput = {};
const backendUrl = "https://charlesfrye--riff.modal.run";

// Setting the default values
function setDefaults() {
  // Set default values
  const defaultPrompt = "2000s techno remix";
  const defaultSeed = 117;
  const defaultDenoising = 0.8;
  const defaultGuidance = 7.0;
  const defaultNumInferenceSteps = 50;
  // Set inferenceInput to default values
  inferenceInput.setPrompt(defaultPrompt);
  inferenceInput.setSeed(defaultSeed);
  inferenceInput.setDenoising(defaultDenoising);
  inferenceInput.setGuidance(defaultGuidance);
  inferenceInput.setNumInferenceSteps(defaultNumInferenceSteps);
  // Set input box values to default values
  document.querySelector("#prompt-input").value = defaultPrompt;
  document.querySelector("#seed-input").value = defaultSeed;
  document.querySelector("#denoising-input").value = defaultDenoising;
  document.querySelector("#guidance-input").value = defaultGuidance;
  document.querySelector("#num-inference-steps-input").value =
    defaultNumInferenceSteps;
}

// Validate user input before sending
function validateInferenceInput() {
  if (!inferenceInput.prompt) {
    alert("Prompt is missing.");
    return false;
  }
  if (!Number.isInteger(inferenceInput.seed)) {
    alert("Seed is missing or not an integer.");
    return false;
  }
  if (
    typeof inferenceInput.denoising !== "number" ||
    inferenceInput.denoising < 0
  ) {
    alert("Denoising value is missing or not a positive number.");
    return false;
  }
  if (
    typeof inferenceInput.guidance !== "number" ||
    inferenceInput.guidance < 0
  ) {
    alert("Guidance value is missing or not a positive number.");
    return false;
  }
  if (
    !Number.isInteger(inferenceInput.numInferenceSteps) ||
    inferenceInput.numInferenceSteps < 0
  ) {
    alert(
      "Number of Inference Steps is missing or not a non-negative integer."
    );
    return false;
  }
  if (!inferenceInput.initAudio) {
    alert("Initial Audio is missing.");
    return false;
  }
  return true;
}
document.addEventListener("DOMContentLoaded", () => {
  setDefaults();
  const button = document.querySelector("#send-to-charles-button");
  const body = document.querySelector("#waveform");
  const audioFileInput = document.querySelector("#audio-file-input");

  document.querySelector("#set-prompt-button").addEventListener("click", () => {
    const prompt = document.querySelector("#prompt-input").value;
    document.querySelector(".prompt-display").innerText = prompt;
  });
  // Loading bar id
  // eslint-disable-next-line no-unused-vars
  let id;

  async function runInference() {
    if (!validateInferenceInput()) {
      alert("Invalid input: Please check your input values.");
      return;
    }


    // Loading Bar
    const loadingBar = document.getElementById("loading-bar");
    let width = 1;
    function frame() {
      if (width >= 100) {
        width = 1;
      } else {
        width++;
        loadingBar.style.width = `${width}%`;
      }
    }
    id = setInterval(frame, 1200);
    try {
      const response = await fetch(backendUrl, {
        method: "POST",
        mode: "cors",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(inferenceInput),
      });

      const responseJson = await response.json();
      const { audioUrl, imageUrl } = responseJson;

      const existingImage = document.getElementById("result-image");
      if (existingImage) {
        existingImage.remove();
      }
      const imageElement = document.createElement("img");
      imageElement.id = "result-image";
      imageElement.src = imageUrl;
      imageElement.alt = "Generated waveform";
      imageElement.width = 75;
      imageElement.height = 75;
      imageElement.style = "margin-top: 2px; margin-bottom: 2px;";
      body.appendChild(imageElement);

      // Get the audio element and remove it if it already exists
      const existingAudio = document.getElementById("result-audio");
      if (existingAudio) {
        existingAudio.remove();
      }
      const audioElement = document.createElement("audio");
      audioElement.id = "result-audio";
      audioElement.src = audioUrl;
      audioElement.type = "audio/mpeg";
      audioElement.controls = true;
      audioElement.style = "margin-top: 2px; margin-bottom: 10px;";

      body.appendChild(audioElement);

      // Stop the loading bar when processing is done
      clearInterval(id);

      document.getElementById("loading-bar").style.width = "0%";
    } catch (err) {
      console.error("An error occurred:", err);
      // Stop the loading bar if an error occurs
      clearInterval(id);

      // Reset loading bar
      document.getElementById("loading-bar").style.width = "0%";
    }
  }

  button.addEventListener("click", () => {
    button.classList.add("bounce");
    setTimeout(() => button.classList.remove("bounce"), 400);

    runInference();
  });

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

    const button = document.querySelector("#set-seed-button");
    button.classList.add("rotate");
    setTimeout(() => button.classList.remove("rotate"), 1000);
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

      const button = document.querySelector("#set-denoising-button");
      button.classList.add("reverse-rotate");
      setTimeout(() => button.classList.remove("reverse-rotate"), 1000);
    });

  document
    .querySelector("#set-guidance-button")
    .addEventListener("click", () => {
      const guidance = parseFloat(
        document.querySelector("#guidance-input").value
      );
      inferenceInput.setGuidance(guidance);

      const button = document.querySelector("#set-guidance-button");
      button.classList.add("rotate");
      setTimeout(() => button.classList.remove("rotate"), 1000);
    });

  document
    .querySelector("#set-num-inference-steps-button")
    .addEventListener("click", () => {
      const numInferenceSteps = Number(
        document.querySelector("#num-inference-steps-input").value
      );
      inferenceInput.setNumInferenceSteps(numInferenceSteps);

      const button = document.querySelector("#set-num-inference-steps-button");
      button.classList.add("reverse-rotate");
      setTimeout(() => button.classList.remove("reverse-rotate"), 1000);
    });
  });
});
