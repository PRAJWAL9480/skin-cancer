async function uploadImage() {
  const input = document.getElementById("imageInput");
  if (!input.files.length) {
    alert("Please select an image");
    return;
  }

  const formData = new FormData();
  formData.append("image", input.files[0]);

  document.getElementById("output").textContent = "Processing...";

  try {
    const response = await fetch("/predict", {
      method: "POST",
      body: formData
    });

    const data = await response.json();
    document.getElementById("output").textContent =
      JSON.stringify(data, null, 2);

  } catch (err) {
    document.getElementById("output").textContent =
      "Error: " + err.message;
  }
}
