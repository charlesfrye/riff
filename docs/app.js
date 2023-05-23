const backendUrl = "https://charlesfrye--riff.modal.run";

document.addEventListener("DOMContentLoaded", () => {
  const button = document.querySelector("button");

  async function runInference() {
    const response = await fetch(backendUrl, {
      method: "POST",
      mode: "cors",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}), // TODO: user input goes here
    });

    const responseJson = await response.json();
    const body = document.querySelector("body");
    // add audio with controls
    const audioElement = document.createElement("audio");
    audioElement.src = responseJson.url;
    audioElement.type = "audio/mpeg";
    audioElement.controls = true;

    body.appendChild(audioElement);
  }
  button.addEventListener("click", runInference);
});
