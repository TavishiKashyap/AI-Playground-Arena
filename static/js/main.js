// ---------- Tab handling ----------
const tabs = document.querySelectorAll("#modeTabs .nav-link");
const panels = {
  detect: document.getElementById("mode-detect"),
  sketch: document.getElementById("mode-sketch"),
  gan: document.getElementById("mode-gan"),
};

tabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    tabs.forEach((t) => t.classList.remove("active"));
    tab.classList.add("active");

    const mode = tab.getAttribute("data-mode");
    for (const key in panels) {
      if (key === mode) panels[key].classList.remove("d-none");
      else panels[key].classList.add("d-none");
    }
  });
});

// ========== 0) Backend detection / simulator ==========
let USE_BACKEND = false;

async function checkBackend() {
  try {
    const controller = new AbortController();
    const id = setTimeout(() => controller.abort(), 1500);
    const resp = await fetch('/api/slots', {signal: controller.signal});
    clearTimeout(id);
    if (resp.ok) { USE_BACKEND = true; console.log('Backend available'); }
  } catch (e) {
    USE_BACKEND = false;
    console.log('Backend not reachable; using client-side simulator');
  }
}

// helper: read a File/Blob to base64
function blobToBase64(blob){
  return new Promise((resolve, reject)=>{
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result.split(',')[1]);
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
}

// wrapper APIs that use backend if available, otherwise simulate locally
async function apiDetectObjects(formData){
  if(USE_BACKEND){
    const resp = await fetch('/api/detect_objects', {method:'POST', body: formData});
    return resp;
  }
  // simulator: return no bboxes and annotated image = uploaded image
  const file = formData.get('image');
  const b64 = await blobToBase64(file);
  return { ok: true, json: async ()=> ({ bboxes: [], annotated_image: b64 }) };
}

async function apiObjectEdit(payload){
  if(USE_BACKEND){
    const resp = await fetch('/api/object_edit', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload)});
    return resp;
  }
  // simulator: simply return the original image
  return { ok: true, json: async ()=> ({ edited_image: payload.image }) };
}

async function apiSketchToImage(formData){
  if(USE_BACKEND){
    const resp = await fetch('/api/sketch_to_image', {method:'POST', body: formData});
    return resp;
  }
  // simulator: use the sketch as output
  const file = formData.get('image');
  const b64 = await blobToBase64(file);
  return { ok: true, json: async ()=> ({ generated_image: b64 }) };
}

async function apiGanGenerate(payload){
  if(USE_BACKEND){
    const resp = await fetch('/api/gan_generate', {
      method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(payload)
    });
    return resp;
  }
  // simulator: generate a procedural canvas image
  const canvas = document.createElement('canvas');
  canvas.width = 128; canvas.height = 128;
  const ctx = canvas.getContext('2d');
  const imgd = ctx.createImageData(canvas.width, canvas.height);
  for(let i=0;i<imgd.data.length;i+=4){
    imgd.data[i] = Math.floor(Math.random()*255);
    imgd.data[i+1] = Math.floor(Math.random()*255);
    imgd.data[i+2] = Math.floor(Math.random()*255);
    imgd.data[i+3] = 255;
  }
  ctx.putImageData(imgd,0,0);
  const dataurl = canvas.toDataURL('image/png').split(',')[1];
  return { ok: true, json: async ()=> ({ generated_image: dataurl }) };
}

// check backend availability on load
checkBackend();

// ========== 1) OBJECT REMOVAL ARENA ==========
let originalImageData = null; // base64
let detectedBboxes = [];      // from backend

const detectInput = document.getElementById("detectImageInput");
const btnRunDetection = document.getElementById("btnRunDetection");
const detectCanvas = document.getElementById("detectCanvas");
const bboxListDiv = document.getElementById("bboxList");
const btnApplyEdits = document.getElementById("btnApplyEdits");
const editedImage = document.getElementById("editedImage");

let detectCtx = detectCanvas.getContext("2d");

btnRunDetection.addEventListener("click", async () => {
  const file = detectInput.files[0];
  if (!file) {
    alert("Please choose an image first.");
    return;
  }

  const formData = new FormData();
  formData.append("image", file);
  const resp = await apiDetectObjects(formData);
  if (!resp.ok) {
    alert("Detection failed");
    return;
  }
  const data = await resp.json();
  detectedBboxes = data.bboxes;

  // Show annotated image
  originalImageData = data.annotated_image;
  drawBase64OnCanvas(detectCanvas, detectCtx, originalImageData);

  // Show bbox list with checkboxes/action dropdowns
  renderBboxList();
});

function renderBboxList() {
  bboxListDiv.innerHTML = "";
  detectedBboxes.forEach((bbox, idx) => {
    const div = document.createElement("div");
    div.className = "bbox-item";
    div.innerHTML = `
      <span>#${idx+1} ${bbox.label} (${bbox.score})</span>
      <select class="form-select form-select-sm bbox-action" data-index="${idx}">
        <option value="keep">Keep</option>
        <option value="remove">Remove</option>
      </select>
    `;
    bboxListDiv.appendChild(div);
  });
}

function drawBase64OnCanvas(canvas, ctx, b64) {
  const img = new Image();
  img.onload = function () {
    canvas.width = img.width;
    canvas.height = img.height;
    ctx.drawImage(img, 0, 0);
  };
  img.src = "data:image/png;base64," + b64;
}

btnApplyEdits.addEventListener("click", async () => {
  if (!originalImageData || detectedBboxes.length === 0) {
    alert("No detections yet.");
    return;
  }

  const actions = [];
  const selects = document.querySelectorAll(".bbox-action");
  selects.forEach((sel) => {
    const idx = parseInt(sel.getAttribute("data-index"));
    const val = sel.value;
    actions.push({
      bbox: detectedBboxes[idx].bbox,
      action: val
    });
  });

  const payload = {
    image: originalImageData,
    actions: actions
  };

  const resp = await apiObjectEdit(payload);
  if (!resp.ok) { alert('Edit failed'); return; }
  const data = await resp.json();
  editedImage.src = "data:image/png;base64," + data.edited_image;
});

// ========== 2) SKETCH TO IMAGE ==========

const sketchCanvas = document.getElementById("sketchCanvas");
const sketchCtx = sketchCanvas.getContext("2d");
sketchCanvas.width = 400;
sketchCanvas.height = 300;
sketchCtx.fillStyle = "#ffffff";
sketchCtx.fillRect(0, 0, sketchCanvas.width, sketchCanvas.height);

let drawing = false;
sketchCanvas.addEventListener("mousedown", () => drawing = true);
sketchCanvas.addEventListener("mouseup", () => drawing = false);
sketchCanvas.addEventListener("mouseleave", () => drawing = false);
sketchCanvas.addEventListener("mousemove", drawSketch);

function drawSketch(e) {
  if (!drawing) return;
  const rect = sketchCanvas.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;
  sketchCtx.fillStyle = "#000000";
  sketchCtx.beginPath();
  sketchCtx.arc(x, y, 3, 0, Math.PI * 2);
  sketchCtx.fill();
}

const btnClearSketch = document.getElementById("btnClearSketch");
const btnGenerateFromSketch = document.getElementById("btnGenerateFromSketch");
const sketchOutput = document.getElementById("sketchOutput");
const guidanceScaleInput = document.getElementById("guidanceScale");
const numStepsInput = document.getElementById("numSteps");

btnClearSketch.addEventListener("click", () => {
  sketchCtx.fillStyle = "#ffffff";
  sketchCtx.fillRect(0, 0, sketchCanvas.width, sketchCanvas.height);
});

btnGenerateFromSketch.addEventListener("click", async () => {
  const guidanceScale = guidanceScaleInput.value;
  const numSteps = numStepsInput.value;

  const blob = await new Promise((resolve) =>
    sketchCanvas.toBlob(resolve, "image/png")
  );

  const formData = new FormData();
  formData.append("image", blob, "sketch.png");
  formData.append("guidance_scale", guidanceScale);
  formData.append("num_steps", numSteps);

  const resp = await apiSketchToImage(formData);
  if (!resp.ok) { alert('Sketch generation failed.'); return; }
  const data = await resp.json();
  sketchOutput.src = "data:image/png;base64," + data.generated_image;
});

// ========== 3) GAN PLAYGROUND ==========

const latentDimInput = document.getElementById("latentDim");
const noiseScaleInput = document.getElementById("noiseScale");
const btnGanGenerate = document.getElementById("btnGanGenerate");
const ganOutput = document.getElementById("ganOutput");

btnGanGenerate.addEventListener("click", async () => {
  const payload = {
    latent_dim: parseInt(latentDimInput.value),
    noise_scale: parseFloat(noiseScaleInput.value)
  };

  const resp = await apiGanGenerate(payload);
  if (!resp.ok) { alert('GAN generation failed'); return; }
  const data = await resp.json();
  ganOutput.src = "data:image/png;base64," + data.generated_image;
});
