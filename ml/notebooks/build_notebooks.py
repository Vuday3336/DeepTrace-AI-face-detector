"""Generate the Phase 2 Kaggle notebooks from plain Python (easier to review/diff than .ipynb JSON).

    python notebooks/build_notebooks.py

The notebooks are thin: every real step is a tested script in ml/scripts, so the notebook and the
command line always do exactly the same thing.
"""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = Path(__file__).resolve().parent

SETUP = r"""# ---- Setup: clone your repo + install the few extras Kaggle doesn't ship ----
import os, subprocess, sys
from pathlib import Path

REPO_URL = "https://github.com/Vuday3336/DeepTrace-AI-face-detector.git"
ON_KAGGLE = Path("/kaggle/working").exists()
WORK = Path("/kaggle/working") if ON_KAGGLE else Path("/content")
REPO = WORK / "deeptrace"
ML = REPO / "ml"

if not REPO.exists():
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(REPO)], check=True)
os.chdir(ML)
print("Working in", Path.cwd())
"""


def md(text: str):
    return new_markdown_cell(text.strip())


def code(text: str):
    return new_code_cell(text.strip())


def data_prep_notebook():
    cells = [
        md("""
# DeepTrace — Phase 2: data preparation & shortcut audit

**Run on:** Kaggle Notebook (free) · Accelerator **GPU T4 x2 or P100** · Internet **On**
**Inputs:** *Add Input* → search **"140k Real and Fake Faces"** (by xhlulu).
Optional cross-generator inputs (section 1b): your generated-faces dataset + a CelebA-HQ dataset.

What this notebook does, in order:
1. Index the 140k dataset into a manifest (hash, size, format, JPEG quality, EXIF) and split `valid` into `val_select` / `val_calib`.
2. **Raw audit** — can file metadata alone tell real from fake? (It must not, after preprocessing.)
3. Near-duplicate removal across splits (pHash + dHash).
4. Canonical preprocessing: strip metadata → MTCNN face crop → 224×224 → JPEG q∈[80,95].
5. **Processed audit** — the same tests again; metadata shortcuts should now be at chance (AUC ≈ 0.5).
6. (Optional) the same pipeline for the cross-generator test set.
7. Package outputs for Phase 3.

**How to run (recommended):** edit `REPO_URL` in the first cell, then **Save Version → Save & Run All
(Commit)**. It runs in the background (up to 12 h) and keeps `/kaggle/working` as the version output.
Running cells interactively works too, but files from an interactive session are not saved unless you
commit. Every script is resumable within a session.
"""),
        code(
            SETUP
            + r"""
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements/kaggle.txt"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements/facenet.txt", "--no-deps"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", ".", "--no-deps"], check=True)

import torch
print("torch", torch.__version__, "| CUDA:", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only (prep will be slow)")
"""
        ),
        code(r"""
# Only needed OUTSIDE Kaggle (e.g. Colab): download the dataset with the Kaggle API.
# Upload kaggle.json first (Kaggle -> Settings -> API -> Create New Token).
if not ON_KAGGLE:
    subprocess.run("pip install -q kaggle && mkdir -p ~/.kaggle && cp /content/kaggle.json ~/.kaggle/ "
                   "&& chmod 600 ~/.kaggle/kaggle.json && kaggle datasets download -d "
                   "xhlulu/140k-real-and-fake-faces -p /content/data --unzip", shell=True, check=True)
SEARCH_ROOT = Path("/kaggle/input") if ON_KAGGLE else Path("/content/data")
PROCESSED = WORK / "processed"
WORKERS = os.cpu_count() or 2
"""),
        md("## 1. Build the 140k manifest"),
        code(r"""
!python scripts/build_manifest_140k.py --search-root {SEARCH_ROOT} --workers {WORKERS}

from deeptrace_ml.data.datasets import find_140k_root
ROOT_140K = find_140k_root(SEARCH_ROOT)
print("140k root:", ROOT_140K)
"""),
        md("""
## 1b. (Optional) Cross-generator test set

Attach as inputs: your **generated faces** (output of `00_generate_diffusion_faces`, saved as a Kaggle
dataset) and a **CelebA-HQ** dataset (Add Input → search "CelebA-HQ"). Then set the paths below.
It is organised here so the dedupe step also checks it against the 140k data; it is prepared and audited in section 6.

⚠️ CelebA-HQ images were created with a neural JPEG-artifact-removal + 4× super-resolution step
(Karras et al., 2018), so these "real" images carry some neural processing. Keep that in mind when a
detector calls them fake — it's documented as a known confound.
"""),
        code(r"""
CELEBAHQ_DIR = None        # e.g. Path("/kaggle/input/<celebahq-dataset>/<images-folder>")
GENERATED_DIR = None       # e.g. Path("/kaggle/input/deeptrace-generated")  (contains sdxl/ and flux-schnell/)
PER_SOURCE = 1000

if CELEBAHQ_DIR and GENERATED_DIR:
    sources = ["--source", f"celebahq:real:{CELEBAHQ_DIR}"]
    for name, folder in (("sdxl", "sdxl"), ("flux_schnell", "flux-schnell")):
        if (GENERATED_DIR / folder).exists():
            sources += ["--source", f"{name}:fake:{GENERATED_DIR / folder}"]
    subprocess.run([sys.executable, "scripts/organize_external.py", "--dataset", "crossgen",
                    "--out-root", str(WORK / "raw/crossgen"), *sources,
                    "--max-per-source", str(PER_SOURCE), "--workers", str(WORKERS)], check=True)
else:
    print("Skipping cross-generator set (paths not set).")
"""),
        md("""
## 2. Raw audit (before any preprocessing)

How to read the results:
- **Metadata classifier AUC** near 0.5 = no shortcut; near 1.0 = the label is leaking through file properties.
- **Pixel-statistics AUC** can legitimately be above 0.5 (generators do leave colour/texture traces), but a very high value warns that a model could win with crude rules like "smoother = fake".
- **Spectra:** look for bright off-centre dots or grid lines (upsampling artifacts) and a bump in the tail of the radial profile.
"""),
        code(r"""
!python scripts/run_audit.py --dataset 140k --manifest data/manifests/140k_raw.csv.gz --root 140k={ROOT_140K} --workers {WORKERS}
"""),
        code(r"""
from IPython.display import Image as ShowImage, display
PLOTS = REPO / "docs/plots/audit/140k"
for name in ["raw_metadata_numeric", "raw_metadata_flags", "raw_pixel_stats", "raw_spectra_1d", "raw_spectra_2d"]:
    print(name); display(ShowImage(filename=str(PLOTS / f"{name}.png")))
"""),
        md("## 3. Near-duplicates across splits"),
        code(r"""
MANIFESTS = ["--manifest", "data/manifests/140k_raw.csv.gz", "--root", f"140k={ROOT_140K}"]
if Path("data/manifests/crossgen_raw.csv.gz").exists():          # created in section 1b
    MANIFESTS += ["--manifest", "data/manifests/crossgen_raw.csv.gz", "--root", f"crossgen={WORK / 'raw/crossgen'}"]
subprocess.run([sys.executable, "scripts/dedupe.py", *MANIFESTS, "--workers", str(WORKERS)], check=True)

import json
print(json.dumps(json.load(open("reports/dedupe_report.json")), indent=2))
dup_plot = REPO / "docs/plots/audit/near_duplicates.png"
if dup_plot.exists():
    display(ShowImage(filename=str(dup_plot)))   # eyeball: are these really the same face?
"""),
        md("""
## 4. Canonical preprocessing (GPU)

MTCNN runs on the GPU in batches. Progress is saved after every batch, so a timeout loses nothing —
rerun the cell. Speed depends on the GPU; the script prints throughput as it goes.
"""),
        code(r"""
!python scripts/prepare_dataset.py --manifest data/manifests/140k_raw.csv.gz --root 140k={ROOT_140K} \
    --duplicates data/manifests/duplicates_drop.csv --out-root {PROCESSED} \
    --device auto --batch-size 128 --io-workers {WORKERS}

print(json.dumps(json.load(open("reports/prepare_140k_raw.json"))["status_counts"], indent=2))
"""),
        md("## 5. Processed audit — did preprocessing remove the shortcuts?"),
        code(r"""
!python scripts/run_audit.py --dataset 140k --manifest data/manifests/140k_raw.csv.gz --root 140k={ROOT_140K} \
    --processed-manifest data/manifests/140k_processed.csv.gz --processed-root {PROCESSED} --workers {WORKERS}

for name in ["processed_status", "processed_metadata_numeric", "processed_pixel_stats", "processed_spectra_1d", "processed_spectra_2d"]:
    print(name); display(ShowImage(filename=str(PLOTS / f"{name}.png")))
"""),
        code(r"""
# Before/after table (the numbers that go into the README — straight from the JSON, never retyped)
import pandas as pd
audit = json.load(open("reports/audit_140k.json"))
rows = []
for stage in ("raw", "processed"):
    for key, res in audit[stage].items():
        if key.startswith("trivial_classifier"):
            rows.append({"stage": stage, "features": res["feature_set"],
                         "logreg_auc": round(res["auc"]["logistic_regression"]["mean"], 3),
                         "boosting_auc": round(res["auc"]["gradient_boosting"]["mean"], 3),
                         "top_features": ", ".join(f["feature"] for f in res["top_features"][:3])})
pd.DataFrame(rows)
"""),
        md("## 6. (Optional) Prepare & audit the cross-generator set"),
        code(r"""
if Path("data/manifests/crossgen_raw.csv.gz").exists():
    CROSS_ROOT = WORK / "raw/crossgen"
    subprocess.run([sys.executable, "scripts/prepare_dataset.py", "--manifest", "data/manifests/crossgen_raw.csv.gz",
                    "--root", f"crossgen={CROSS_ROOT}", "--duplicates", "data/manifests/duplicates_drop.csv",
                    "--out-root", str(PROCESSED), "--device", "auto", "--batch-size", "32"], check=True)
    subprocess.run([sys.executable, "scripts/run_audit.py", "--dataset", "crossgen",
                    "--manifest", "data/manifests/crossgen_raw.csv.gz", "--root", f"crossgen={CROSS_ROOT}",
                    "--processed-manifest", "data/manifests/crossgen_processed.csv.gz",
                    "--processed-root", str(PROCESSED), "--sample-per-group", "1000",
                    "--spectrum-per-group", "500", "--workers", str(WORKERS)], check=True)
    CROSS_PLOTS = REPO / "docs/plots/audit/crossgen"
    for name in ["raw_metadata_numeric", "processed_status", "processed_spectra_1d"]:
        print(name); display(ShowImage(filename=str(CROSS_PLOTS / f"{name}.png")))
else:
    print("No cross-generator manifest — skipped.")
"""),
        md("""
## 7. Save outputs

1. If you ran interactively, commit with **Save Version → Save & Run All** (only committed runs keep files).
2. Open the finished version → **Output** → **New Dataset**, name it `deeptrace-processed`.
   Phase 3 training attaches this dataset (no re-processing needed).
3. Download `deeptrace_phase2_artifacts.zip` (below) and commit its manifests, reports and plots to
   your repo — those are small and make the audit reproducible and reviewable on GitHub.
"""),
        code(r"""
import shutil
bundle = WORK / "deeptrace_phase2_artifacts"
if bundle.exists():
    shutil.rmtree(bundle)
shutil.copytree(ML / "data/manifests", bundle / "ml/data/manifests")
shutil.copytree(ML / "reports", bundle / "ml/reports")
shutil.copytree(REPO / "docs/plots", bundle / "docs/plots")
print(shutil.make_archive(str(bundle), "zip", bundle))
"""),
    ]
    return new_notebook(cells=cells, metadata=_metadata())


def generation_notebook():
    cells = [
        md("""
# DeepTrace — generate diffusion faces (cross-generator TEST set)

**Run on:** Kaggle Notebook (free) · Accelerator **GPU T4 x2 or P100** · Internet **On**

Models (both free & open):
- **SDXL base 1.0** — CreativeML Open RAIL++-M license.
- **FLUX.1 [schnell]** — Apache-2.0. Much larger download (~34 GB); loaded 4-bit to fit 16 GB.

These images are **never used for training** — they measure whether a detector trained on StyleGAN
generalises to modern generators.

**How to run (recommended):** set `MODELS` below, then **Save Version → Save & Run All (Commit)**.
It runs in the background (Kaggle's per-run limit is 12 h) and everything in `/kaggle/working` is kept
as the version's output. Each model runs in its own process, so GPU memory is freed between them.

**Splitting across runs** (free GPU quota is weekly): generation is resumable. Attach the previous
version's output as an input, set `PREVIOUS_OUTPUT` to it, and commit again — existing images are
copied in and generation continues from the next index.

If a Hugging Face model page asks you to accept terms, accept it on the website, then add your HF
token as a Kaggle secret named `HF_TOKEN` (Add-ons → Secrets).
"""),
        code(
            SETUP
            + r"""
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements/generation.txt"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", ".", "--no-deps"], check=True)

try:  # optional Hugging Face login from a Kaggle secret
    from kaggle_secrets import UserSecretsClient
    from huggingface_hub import login
    login(token=UserSecretsClient().get_secret("HF_TOKEN"))
    print("Logged in to Hugging Face")
except Exception as exc:  # no secret configured -> public models still work
    print("No HF login:", type(exc).__name__)

OUT = WORK / "generated"

# ---- Parameters ----
MODELS = ["sdxl"]            # ["sdxl"], ["flux-schnell"], or both (both may not fit in one 12 h run)
COUNT_PER_MODEL = 700        # a bit above 600 leaves room for images where MTCNN finds no face
PREVIOUS_OUTPUT = None       # e.g. Path("/kaggle/input/<previous-version-output>/generated")

import shutil
if PREVIOUS_OUTPUT is not None:
    shutil.copytree(PREVIOUS_OUTPUT, OUT, dirs_exist_ok=True)
    print("Resumed from", PREVIOUS_OUTPUT)
"""
        ),
        md("## Generate"),
        code(r"""
for model in MODELS:
    subprocess.run([sys.executable, "scripts/generate_diffusion_faces.py", "--model", model,
                    "--count", str(COUNT_PER_MODEL), "--out", str(OUT)], check=True)
    # If FLUX outputs are black/degenerate, add "--flux-dtype", "bfloat16" above and rerun.
"""),
        code(r"""
# Contact sheet of the first images — check they look like photos of single faces
from PIL import Image
from IPython.display import display

def contact_sheet(folder, n=12, size=192):
    files = sorted(Path(folder).glob("*.png"))[:n]
    sheet = Image.new("RGB", (size * 6, size * ((len(files) + 5) // 6)))
    for i, f in enumerate(files):
        sheet.paste(Image.open(f).convert("RGB").resize((size, size)), ((i % 6) * size, (i // 6) * size))
    return sheet

for model in MODELS:
    if (OUT / model).exists():
        print(model); display(contact_sheet(OUT / model))
"""),
        md("""
## Save as a dataset

When the committed run finishes, open the version → **Output** → **New Dataset**, named
`deeptrace-generated` (it should contain `sdxl/` and/or `flux-schnell/`). Attach it in
`01_data_prep_and_audit` (section 1b). The `metadata.jsonl` files (prompt, seed, settings per image)
make every image reproducible, and also record how many outputs were degenerate.
"""),
    ]
    return new_notebook(cells=cells, metadata=_metadata())


LOCATE = r"""
# ---- Locate Phase 2 outputs among the attached inputs ----
import os, json, shutil
from pathlib import Path

def find_dir(root, predicate, max_depth=6):
    base = len(Path(root).parts)
    for dirpath, dirnames, _ in os.walk(root):
        p = Path(dirpath)
        if len(p.parts) - base >= max_depth:
            dirnames.clear(); continue
        if predicate(p):
            return p
    return None

INPUT = Path("/kaggle/input") if ON_KAGGLE else Path("/content/input")
PROCESSED = find_dir(INPUT, lambda p: p.name == "processed" and (p / "140k").is_dir())
MANIFESTS = find_dir(INPUT, lambda p: (p / "140k_processed.csv.gz").is_file())
if PROCESSED is None or MANIFESTS is None:
    raise FileNotFoundError("Attach the 'deeptrace-processed' dataset (output of 01_data_prep_and_audit).")
RAW_140K = find_dir(INPUT, lambda p: p.name == "real-vs-fake" and (p / "test").is_dir())
RAW_CROSSGEN = find_dir(INPUT, lambda p: p.name == "crossgen" and (p / "fake").is_dir())
PROCESSED_MANIFESTS = [str(m) for m in (MANIFESTS / "140k_processed.csv.gz", MANIFESTS / "crossgen_processed.csv.gz") if m.is_file()]
print("processed crops:", PROCESSED)
print("manifests:", PROCESSED_MANIFESTS)
print("raw 140k:", RAW_140K, "| raw crossgen:", RAW_CROSSGEN)
"""

INSTALL_TRAIN = r"""
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements/kaggle.txt", "-r", "requirements/train.txt"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements/facenet.txt", "--no-deps"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", ".", "--no-deps"], check=True)
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

import torch
print("torch", torch.__version__, "| GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")

def run(*args):
    print("$", " ".join(map(str, args)))
    subprocess.run([sys.executable, *map(str, args)], check=True)
"""


def train_notebook():
    cells = [
        md("""
# DeepTrace — Phases 3 & 4: train and evaluate a model

**Run on:** Kaggle (free) · GPU **T4 x2 or P100** · Internet **On**
**Inputs:** `deeptrace-processed` (output of `01_data_prep_and_audit`).

Run this notebook once per model: `effnet_b0` (baseline), `convnext_tiny`, `clip_probe`, and optionally
`vit_small`. Use **Save Version → Save & Run All** so it runs in the background and keeps the outputs.

1. **Train** on the 140k `train` split, early-stopping on `val_select` ROC-AUC (CLIP: fit a linear probe).
2. **Predict** on `val_calib`, `test` and (if present) the cross-generator set.
3. **Calibrate** temperature, threshold and uncertainty band on `val_calib` only.
4. **Evaluate** every test set with bootstrap 95% CIs.

Seeds: first run `SEEDS = [42]` for every model. For the finalist(s), rerun with `[42, 43, 44]` and report
mean ± std across seeds. Evaluation below uses the first seed's checkpoint.
"""),
        code(SETUP + INSTALL_TRAIN),
        code(LOCATE),
        code("""
MODEL = "effnet_b0"          # effnet_b0 | convnext_tiny | vit_small | clip_probe
SEEDS = [42]
RUNS = WORK / "runs"
REPORTS = ML / "reports"
"""),
        md("## 1. Train"),
        code("""
for seed in SEEDS:
    run_dir = RUNS / f"{MODEL}_seed{seed}"
    if MODEL == "clip_probe":
        run("scripts/train_clip_probe.py", "--manifest", MANIFESTS / "140k_processed.csv.gz",
            "--data-root", PROCESSED, "--run-dir", run_dir, "--seed", seed)
    else:
        run("scripts/train.py", "--config", f"configs/train/{MODEL}.yaml",
            "--manifest", MANIFESTS / "140k_processed.csv.gz", "--data-root", PROCESSED,
            "--run-dir", run_dir, "--seed", seed)
    print(json.load(open(run_dir / "final_metrics.json")))
CHECKPOINT = RUNS / f"{MODEL}_seed{SEEDS[0]}" / "best.pt"
"""),
        code("""
# Learning curves (fine-tuned models only)
import pandas as pd
import matplotlib.pyplot as plt
history_file = RUNS / f"{MODEL}_seed{SEEDS[0]}" / "metrics_epoch.jsonl"
if history_file.exists():
    h = pd.read_json(history_file, lines=True)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    h.plot(x="epoch", y=["train_loss", "val_loss"], ax=ax[0], marker="o"); ax[0].set_title("loss")
    h.plot(x="epoch", y="val_auc", ax=ax[1], marker="o"); ax[1].set_title("val_select ROC-AUC")
    plt.show()
"""),
        md("## 2–4. Predict → calibrate (val_calib only) → evaluate"),
        code("""
PRED = REPORTS / f"predictions_{MODEL}.csv"
manifest_args = [a for m in PROCESSED_MANIFESTS for a in ("--manifest", m)]
run("scripts/predict.py", "--checkpoint", CHECKPOINT, *manifest_args, "--data-root", PROCESSED,
    "--splits", "val_calib", "test", "crossgen", "own", "--out", PRED)
run("scripts/calibrate.py", "--predictions", PRED, "--model-name", MODEL)
run("scripts/evaluate.py", "--predictions", PRED, "--calibration", REPORTS / f"calibration_{MODEL}.json",
    "--model-name", MODEL)
"""),
        code("""
report = json.load(open(REPORTS / f"eval_{MODEL}.json"))
rows = []
for name, m in report["test_sets"].items():
    t = m["at_tuned_threshold"]
    ci = m["ci95"]["roc_auc"]
    rows.append({"test_set": name, "n": m["n"], "roc_auc": m["roc_auc"],
                 "auc_ci95": f"[{ci['low']:.3f}, {ci['high']:.3f}]" if ci else None,
                 "bal_acc": t["balanced_accuracy"], "fpr": t["fpr"], "tpr": t["recall_tpr"], "ece": m["ece_15"]})
pd.DataFrame(rows).round(4)
"""),
        md("""
## Save

When the committed run finishes: **Output → New Dataset** named `deeptrace-runs-<model>`. Notebook
`03_robustness_explain_export` attaches these. Also copy `ml/reports/*.json` and `docs/plots/eval/` into
your repo — they are the only source of the numbers in the README.
"""),
        code("""
bundle = WORK / f"deeptrace_{MODEL}_artifacts"
shutil.rmtree(bundle, ignore_errors=True)
shutil.copytree(REPORTS, bundle / "ml/reports")
shutil.copytree(REPO / "docs/plots", bundle / "docs/plots", dirs_exist_ok=True)
print(shutil.make_archive(str(bundle), "zip", bundle))
"""),
    ]
    return new_notebook(cells=cells, metadata=_metadata())


def evaluate_notebook():
    cells = [
        md("""
# DeepTrace — Phases 5 & 6: robustness, explainability, export, model selection

**Run on:** Kaggle (free) · GPU on · Internet **On**
**Inputs:** `deeptrace-processed`, the `140k Real and Fake Faces` dataset (raw images for robustness), and
every `deeptrace-runs-<model>` dataset from `02_train_and_evaluate`.

1. **Robustness** — JPEG 100→30, downscaling, blur, screenshot chains, through the serving pipeline.
2. **Explainability** — Grad-CAM / Grad-CAM++ (CNN), Grad-CAM + attention rollout (ViT), worst-error
   galleries, and the model-randomisation sanity check.
3. **FFT analysis** of real vs fake crops.
4. **Export** each model to ONNX (fails loudly on any parity mismatch) and **benchmark CPU latency**.
5. **Select** the deployment model with the pre-registered rule and render `docs/METRICS.md`.
"""),
        code(SETUP + INSTALL_TRAIN),
        code(LOCATE),
        code("""
CHECKPOINTS = {}
REPORTS = ML / "reports"
for model in ("effnet_b0", "convnext_tiny", "vit_small", "clip_probe"):
    found = find_dir(INPUT, lambda p, m=model: p.name.startswith(f"{m}_seed") and (p / "best.pt").is_file())
    reports = find_dir(INPUT, lambda p, m=model: (p / f"calibration_{m}.json").is_file())
    if found and reports:
        CHECKPOINTS[model] = found / "best.pt"
        for f in reports.glob(f"*_{model}.*"):
            shutil.copy(f, REPORTS / f.name)
print("models found:", {k: str(v) for k, v in CHECKPOINTS.items()})
VERSION = "1.0.0"
"""),
        md("## 1. Robustness"),
        code("""
roots = ["--root", f"140k={RAW_140K}"] + (["--root", f"crossgen={RAW_CROSSGEN}"] if RAW_CROSSGEN else [])
pm_args = [a for m in PROCESSED_MANIFESTS for a in ("--processed-manifest", m)]
for model, ckpt in CHECKPOINTS.items():
    run("scripts/robustness.py", "--checkpoint", ckpt, "--model-name", model, *pm_args, *roots,
        "--calibration", REPORTS / f"calibration_{model}.json")
"""),
        md("## 2. Explainability galleries + sanity check"),
        code("""
from IPython.display import Image as ShowImage, display
for model, ckpt in CHECKPOINTS.items():
    run("scripts/explain.py", "--checkpoint", ckpt, "--model-name", model,
        "--predictions", REPORTS / f"predictions_{model}.csv", "--calibration", REPORTS / f"calibration_{model}.json",
        "--data-root", PROCESSED, "--device", "cuda" if torch.cuda.is_available() else "cpu")
    print(model, json.load(open(REPORTS / f"explain_{model}.json")).get("sanity_check"))
    for png in sorted((REPO / f"docs/plots/explain/{model}").glob("*.png")):
        print(png.name); display(ShowImage(filename=str(png)))
"""),
        md("""
**Write your interpretation in `docs/EXPLAINABILITY.md`** as observations, not claims about what the model
"understands" (e.g. *"in 8 sampled false positives on crossgen, heat concentrates on hair edges"*). If the
sanity-check correlation is high, state that the heatmaps are unreliable for that model.
"""),
        md("## 3. Frequency analysis"),
        code("""
run("scripts/fft_analysis.py", *[a for m in PROCESSED_MANIFESTS for a in ("--processed-manifest", m)],
    "--data-root", PROCESSED)
for name in ("radial_profiles", "difference_vs_reference"):
    f = REPO / f"docs/plots/fft/{name}.png"
    if f.exists():
        display(ShowImage(filename=str(f)))
"""),
        md("## 4. Export to ONNX + CPU latency"),
        code("""
ARTIFACTS = WORK / "artifacts"
for model, ckpt in CHECKPOINTS.items():
    out = ARTIFACTS / model
    run("scripts/export_model.py", "--checkpoint", ckpt, "--calibration", REPORTS / f"calibration_{model}.json",
        "--eval-report", REPORTS / f"eval_{model}.json", "--robustness-report", REPORTS / f"robustness_{model}.json",
        "--version", VERSION, "--out-dir", out)
    # 2 threads approximates the free Hugging Face Spaces CPU; Kaggle CPUs differ, so treat as indicative.
    run("scripts/benchmark_latency.py", "--bundle-dir", out, "--threads", "2", "--checkpoint", ckpt)
"""),
        md("## 5. Model selection (pre-registered rule) + metrics report"),
        code("""
evals = [a for m in CHECKPOINTS for a in ("--eval", REPORTS / f"eval_{m}.json")]
lats = [a for m in CHECKPOINTS for a in ("--latency", REPORTS / f"latency_{m}.json")]
run("scripts/compare_models.py", *evals, *lats)
print(json.dumps(json.load(open(REPORTS / "model_selection.json"))["decision"], indent=2))
run("scripts/render_metrics_md.py")
print(open(REPO / "docs/METRICS.md").read()[:3000])
"""),
        md("""
## Save & publish

1. Commit the run (**Save & Run All**), download `deeptrace_final_artifacts.zip`, and copy `ml/reports`,
   `docs/plots` and `docs/METRICS.md` into your repo.
2. Upload the **selected** bundle to the Hugging Face Hub (free), locally or with a Kaggle secret `HF_TOKEN`:
   `python scripts/upload_model_to_hub.py --bundle-dir artifacts/<winner> --repo-id <user>/deeptrace-model`
"""),
        code("""
bundle = WORK / "deeptrace_final_artifacts"
shutil.rmtree(bundle, ignore_errors=True)
shutil.copytree(REPORTS, bundle / "ml/reports")
shutil.copytree(REPO / "docs/plots", bundle / "docs/plots", dirs_exist_ok=True)
shutil.copy(REPO / "docs/METRICS.md", bundle / "docs/METRICS.md")
shutil.copytree(ARTIFACTS, bundle / "artifacts")
print(shutil.make_archive(str(bundle), "zip", bundle))
"""),
    ]
    return new_notebook(cells=cells, metadata=_metadata())


def _metadata() -> dict:
    return {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
    }


def main() -> None:
    for name, nb in (
        ("00_generate_diffusion_faces.ipynb", generation_notebook()),
        ("01_data_prep_and_audit.ipynb", data_prep_notebook()),
        ("02_train_and_evaluate.ipynb", train_notebook()),
        ("03_robustness_explain_export.ipynb", evaluate_notebook()),
    ):
        nbformat.validate(nb)
        path = HERE / name
        nbformat.write(nb, path)
        print("wrote", path)


if __name__ == "__main__":
    main()
