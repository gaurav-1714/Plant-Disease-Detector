const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const preview = document.getElementById("preview");
const dropzoneContent = document.getElementById("dropzoneContent");
const analyzeBtn = document.getElementById("analyzeBtn");
const loader = document.getElementById("loader");
const errorBox = document.getElementById("errorBox");
const resultCard = document.getElementById("resultCard");

const uploadTabBtn = document.getElementById("uploadTabBtn");
const cameraTabBtn = document.getElementById("cameraTabBtn");
const uploadMode = document.getElementById("uploadMode");
const cameraMode = document.getElementById("cameraMode");

const cameraVideo = document.getElementById("cameraVideo");
const cameraCanvas = document.getElementById("cameraCanvas");
const capturedPreview = document.getElementById("capturedPreview");
const cameraPlaceholder = document.getElementById("cameraPlaceholder");
const startCameraBtn = document.getElementById("startCameraBtn");
const captureBtn = document.getElementById("captureBtn");
const retakeBtn = document.getElementById("retakeBtn");
const switchCameraBtn = document.getElementById("switchCameraBtn");

let selectedFile = null;   // Blob/File to send to the API, from either source
let cameraStream = null;
let facingMode = "environment"; // rear camera by default (best for phones)

// ── Mode tabs ────────────────────────────────────────────────────────────
uploadTabBtn.addEventListener("click", () => switchMode("upload"));
cameraTabBtn.addEventListener("click", () => switchMode("camera"));

function switchMode(mode) {
  const toUpload = mode === "upload";
  uploadTabBtn.classList.toggle("active", toUpload);
  cameraTabBtn.classList.toggle("active", !toUpload);
  uploadMode.hidden = !toUpload;
  cameraMode.hidden = toUpload;

  if (toUpload) {
    stopCamera();
  }
  hideError();
  resultCard.hidden = true;
  selectedFile = null;
  analyzeBtn.disabled = true;
}

// ── Upload dropzone (unchanged behavior) ─────────────────────────────────
dropzone.addEventListener("click", () => fileInput.click());

dropzone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropzone.classList.add("dragover");
});
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("dragover");
  if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});

fileInput.addEventListener("change", () => {
  if (fileInput.files.length) handleFile(fileInput.files[0]);
});

function handleFile(file) {
  hideError();
  const allowed = ["image/jpeg", "image/png", "image/jpg", "image/webp"];
  if (!allowed.includes(file.type)) {
    showError("Please upload a JPEG, PNG, or WEBP image.");
    return;
  }
  if (file.size > 10 * 1024 * 1024) {
    showError("File too large. Maximum size is 10MB.");
    return;
  }
  selectedFile = file;
  const reader = new FileReader();
  reader.onload = (e) => {
    preview.src = e.target.result;
    preview.hidden = false;
    dropzoneContent.hidden = true;
  };
  reader.readAsDataURL(file);
  analyzeBtn.disabled = false;
  resultCard.hidden = true;
}

// ── Camera capture ────────────────────────────────────────────────────────
startCameraBtn.addEventListener("click", startCamera);
captureBtn.addEventListener("click", captureFrame);
retakeBtn.addEventListener("click", retake);
switchCameraBtn.addEventListener("click", () => {
  facingMode = facingMode === "environment" ? "user" : "environment";
  startCamera();
});

async function startCamera() {
  hideError();
  stopCamera();

  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    showError("Camera access isn't supported in this browser. Try Chrome or Safari, and make sure you're on HTTPS (or localhost).");
    return;
  }

  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: facingMode }, width: { ideal: 1280 }, height: { ideal: 1280 } },
      audio: false,
    });
    cameraVideo.srcObject = cameraStream;
    cameraVideo.hidden = false;
    cameraPlaceholder.hidden = true;
    capturedPreview.hidden = true;

    startCameraBtn.hidden = true;
    captureBtn.hidden = false;
    retakeBtn.hidden = true;
    switchCameraBtn.hidden = false;
  } catch (err) {
    if (err.name === "NotAllowedError") {
      showError("Camera permission was denied. Allow camera access in your browser settings and try again.");
    } else if (err.name === "NotFoundError") {
      showError("No camera was found on this device.");
    } else {
      showError("Couldn't access the camera: " + err.message + ". Note: camera access requires HTTPS or localhost.");
    }
  }
}

function captureFrame() {
  const w = cameraVideo.videoWidth;
  const h = cameraVideo.videoHeight;
  cameraCanvas.width = w;
  cameraCanvas.height = h;
  const ctx = cameraCanvas.getContext("2d");
  ctx.drawImage(cameraVideo, 0, 0, w, h);

  cameraCanvas.toBlob((blob) => {
    selectedFile = new File([blob], "camera-capture.jpg", { type: "image/jpeg" });
    capturedPreview.src = URL.createObjectURL(blob);
    capturedPreview.hidden = false;
    cameraVideo.hidden = true;

    captureBtn.hidden = true;
    retakeBtn.hidden = false;
    switchCameraBtn.hidden = true;

    analyzeBtn.disabled = false;
    resultCard.hidden = true;
    stopCamera(); // free the camera once we have our frame
  }, "image/jpeg", 0.92);
}

function retake() {
  capturedPreview.hidden = true;
  selectedFile = null;
  analyzeBtn.disabled = true;
  startCamera();
}

function stopCamera() {
  if (cameraStream) {
    cameraStream.getTracks().forEach((track) => track.stop());
    cameraStream = null;
  }
}

// Clean up the camera if the user navigates away
window.addEventListener("beforeunload", stopCamera);


// ── Analyze ──────────────────────────────────────────────────────────────
analyzeBtn.addEventListener("click", async () => {
  if (!selectedFile) return;
  hideError();
  loader.hidden = false;
  analyzeBtn.disabled = true;
  resultCard.hidden = true;

  try {
    const formData = new FormData();
    formData.append("file", selectedFile);

    const resp = await fetch(`${API_BASE_URL}/predict`, {
      method: "POST",
      body: formData,
    });

    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ detail: "Prediction failed." }));
      throw new Error(err.detail || "Prediction failed.");
    }

    const data = await resp.json();
    renderResult(data);
  } catch (err) {
    showError(err.message || "Something went wrong. Is the API running?");
  } finally {
    loader.hidden = true;
    analyzeBtn.disabled = false;
  }
});

function severityClass(severity) {
  const map = {
    None: "severity-none",
    Low: "severity-low",
    Medium: "severity-medium",
    High: "severity-high",
    Critical: "severity-critical",
  };
  return map[severity] || "severity-medium";
}

function renderResult(data) {
  document.getElementById("resultDisease").textContent = data.disease;
  document.getElementById("resultCrop").textContent = `${data.crop}${data.is_healthy ? " • Healthy" : ""}`;
  document.getElementById("resultDescription").textContent = data.description;

  const badge = document.getElementById("severityBadge");
  badge.textContent = data.severity;
  badge.className = `severity-badge ${severityClass(data.severity)}`;

  const pct = Math.round(data.confidence * 100);
  document.getElementById("confidenceFill").style.width = `${pct}%`;
  document.getElementById("confidenceValue").textContent = `${pct}%`;

  fillList("symptomsList", data.symptoms, "No specific symptoms.");
  fillList("treatmentList", data.treatment, "No treatment needed.");
  fillList("preventionList", data.prevention, "No prevention tips available.");

  const top3El = document.getElementById("top3List");
  top3El.innerHTML = "";
  data.top3.forEach((item) => {
    const div = document.createElement("div");
    div.className = "top3-item";
    div.innerHTML = `<span>${formatClassName(item.class)}</span><span>${Math.round(item.confidence * 100)}%</span>`;
    top3El.appendChild(div);
  });

  document.getElementById("inferenceTime").textContent = data.inference_ms;

  resultCard.hidden = false;
  resultCard.scrollIntoView({ behavior: "smooth", block: "start" });
}

function fillList(id, items, emptyText) {
  const el = document.getElementById(id);
  el.innerHTML = "";
  if (!items || items.length === 0) {
    const li = document.createElement("li");
    li.textContent = emptyText;
    el.appendChild(li);
    return;
  }
  items.forEach((text) => {
    const li = document.createElement("li");
    li.textContent = text;
    el.appendChild(li);
  });
}

function formatClassName(name) {
  return name.replace(/___/g, " — ").replace(/_/g, " ");
}

function showError(msg) {
  errorBox.textContent = msg;
  errorBox.hidden = false;
}
function hideError() {
  errorBox.hidden = true;
}

// ── Stats + API status ───────────────────────────────────────────────────
async function loadStats() {
  try {
    const resp = await fetch(`${API_BASE_URL}/stats`);
    const data = await resp.json();
    document.getElementById("statClasses").textContent = data.total_classes;
    document.getElementById("statCrops").textContent = data.supported_crops;
    document.getElementById("statAccuracy").textContent = data.target_accuracy;
    document.getElementById("statSpeed").textContent = data.avg_inference_ms;
    setApiStatus(true);
  } catch {
    setApiStatus(false);
  }
}

function setApiStatus(online) {
  const el = document.getElementById("api-status");
  el.textContent = online ? "online" : "offline";
  el.style.color = online ? "#3f9e60" : "#c0392b";
}

loadStats();
