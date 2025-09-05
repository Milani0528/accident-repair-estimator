// -----------------------
// CONFIG
// -----------------------
const API_BASE = "http://localhost:5000";   // Flask backend

const CLASSES = [
    "Front_View",
    "Non_Front_View",
    "Non_Rear_Bumper",
    "Non_Sedan_Side_View",
    "Rear_Bumper",
    "Sedan_Side_View",
];

function isAccident(label) {
    return ["Front_View", "Rear_Bumper", "Sedan_Side_View"].includes(label);
}

// -----------------------
// ELEMENTS
// -----------------------
const $file   = document.getElementById("file");
const $btn    = document.getElementById("btn");
const $img    = document.getElementById("img");
const $error  = document.getElementById("error");
const $result = document.getElementById("result");
const $rLabel = document.getElementById("r-label");
const $rConf  = document.getElementById("r-conf");
const $rEst   = document.getElementById("r-est");
const $rNote  = document.getElementById("r-note");
const $rModel = document.getElementById("r-model");
const $loading= document.getElementById("loading");

let currentFile = null;

// -----------------------
// EVENTS
// -----------------------
$file.addEventListener("change", (e) => {
    $error.style.display = "none";
    $result.style.display = "none";
    currentFile = e.target.files?.[0] || null;
    $btn.disabled = !currentFile;
    if (currentFile) {
        $img.src = URL.createObjectURL(currentFile);
        $img.style.display = "block";
    } else {
        $img.style.display = "none";
    }
});

document.getElementById("form").addEventListener("submit", async (e) => {
    e.preventDefault();
    if (!currentFile) return;
    setBusy(true);
    try {
        const data = await analyze(currentFile);
        showResult(data);
    } catch (err) {
        $error.textContent = err.message || "Unexpected error";
        $error.style.display = "block";
    } finally {
        setBusy(false);
    }
});

// -----------------------
// HELPERS
// -----------------------
function setBusy(isBusy) {
    $loading.style.display = isBusy ? "flex" : "none";
    $btn.disabled = isBusy || !currentFile;
}

async function analyze(file) {
    const form = new FormData();
    form.append("image", file);

    const res = await fetch(`${API_BASE}/api/predict`, {
        method: "POST",
        body: form
    });

    if (!res.ok) {
        let errText = `HTTP ${res.status}`;
        try { const j = await res.json(); if (j.error) errText = j.error; } catch {}
        throw new Error(errText);
    }
    return res.json();
}

function showResult(r) {
    const badge = isAccident(r.label)
        ? `<span class="badge red">Accident</span>`
        : `<span class="badge green">No Damage</span>`;

    $rLabel.innerHTML = `${r.label} ${badge}`;
    $rConf.textContent  = (typeof r.confidence === "number") ? r.confidence.toFixed(3) : "-";
    const est = (typeof r.estimate_lkr === "number") ? r.estimate_lkr.toLocaleString() : "-";
    $rEst.textContent   = `${est} ${r.currency || "LKR"}`;
    $rNote.textContent  = r.message || "";
    $rModel.textContent = r.model_status || "";
    $result.style.display = "block";
}
