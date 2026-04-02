"""
app.py  —  Fetal Brain MRI Segmentation Demo
"""

from __future__ import annotations

import sys
from pathlib import Path

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataloader import TISSUE_LABELS, load_config
from src.models.unet3d import build_unet3d
from src.preprocessing.transforms import PreprocessingPipeline
from src.evaluation.metrics import compute_multiclass_dice

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

CONFIG_PATH = ROOT / "configs" / "config.yaml"
CHECKPOINT  = ROOT / "outputs" / "checkpoint_best.pth"
NUM_CLASSES = 8
CMAP        = plt.cm.get_cmap("tab10", NUM_CLASSES)

# ✅ Handles the doubled feta_2.4/feta_2.4 folder and full BIDS layout
def _find_bids_root() -> Path | None:
    """
    Find the BIDS root — the folder that directly contains sub-XXX subject folders.
    Walks up to 4 levels deep under ROOT/data to find it.
    """
    data_root = ROOT / "data"
    if not data_root.exists():
        print(f"⚠  'data/' folder not found at {data_root}")
        return None

    # Search up to depth 4 for a folder that contains sub-XXX subfolders
    for candidate in sorted(data_root.rglob("*")):
        if not candidate.is_dir():
            continue
        # Skip derivatives — those are label/biometry subdirs, not the BIDS root
        if "derivatives" in candidate.parts:
            continue
        # A valid BIDS root has at least one sub-XXX child directory
        sub_dirs = [d for d in candidate.iterdir()
                    if d.is_dir() and d.name.startswith("sub-")]
        if sub_dirs:
            print(f"✓ BIDS root found: {candidate}")
            print(f"  Subject folders: {[d.name for d in sorted(sub_dirs)[:5]]}{'...' if len(sub_dirs) > 5 else ''}")
            return candidate

    print("⚠  Could not find a BIDS root with sub-XXX folders under data/")
    return None


def _find_labels_root(bids_root: Path) -> Path | None:
    """
    Find the derivatives/labels folder that holds dseg files.
    Tries common FeTA label derivative locations.
    """
    candidates = [
        bids_root / "derivatives" / "labels",
        bids_root / "derivatives" / "manual_masks",
        bids_root / "derivatives" / "seg",
        bids_root / "derivatives",          # flat derivatives
        bids_root,                           # labels co-located with images
    ]
    for c in candidates:
        if c.exists():
            # Verify it actually has dseg files somewhere underneath
            if any(c.rglob("*dseg*.nii.gz")):
                print(f"✓ Labels root found: {c}")
                return c
    # Last resort — search the whole bids parent tree for dseg files
    parent = bids_root.parent
    if any(parent.rglob("*dseg*.nii.gz")):
        print(f"✓ Labels root (parent fallback): {parent}")
        return parent
    print("⚠  Could not find labels/dseg files")
    return None


BIDS_ROOT   = _find_bids_root()
LABELS_ROOT = _find_labels_root(BIDS_ROOT) if BIDS_ROOT else None

# ---------------------------------------------------------------------------
# Subject discovery
# ---------------------------------------------------------------------------

def _get_subject_list() -> list[str]:
    if BIDS_ROOT is None:
        return []
    sub_dirs = sorted(
        d.name for d in BIDS_ROOT.iterdir()
        if d.is_dir() and d.name.startswith("sub-")
    )
    print(f"✓ {len(sub_dirs)} subjects found: {sub_dirs[:5]}{'...' if len(sub_dirs) > 5 else ''}")
    return sub_dirs


def _get_image_path(subject_id: str) -> Path | None:
    """
    Find the T2w image for a subject.
    Looks for:  <bids_root>/<subject_id>/anat/<subject_id>*T2w*.nii.gz
    Falls back to any non-dseg .nii.gz under that subject folder.
    """
    if BIDS_ROOT is None:
        return None
    sub_dir = BIDS_ROOT / subject_id
    if not sub_dir.exists():
        return None

    # Priority 1: standard T2w file
    for f in sorted(sub_dir.rglob("*.nii.gz")):
        if "T2w" in f.name and "dseg" not in f.name:
            return f
    # Priority 2: any non-label nii.gz under this subject
    for f in sorted(sub_dir.rglob("*.nii.gz")):
        if "dseg" not in f.name and "label" not in f.name.lower():
            return f
    return None


def _get_label_path(subject_id: str) -> Path | None:
    """
    Find the dseg label for a subject anywhere under LABELS_ROOT.
    """
    if LABELS_ROOT is None:
        return None
    for f in sorted(LABELS_ROOT.rglob("*.nii.gz")):
        if subject_id in str(f) and "dseg" in f.name:
            return f
    return None


SUBJECT_LIST = _get_subject_list()

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

_model  = None
_config = None

def _load_model():
    global _model, _config
    if _model is not None:
        return _model, _config

    _config = load_config(str(CONFIG_PATH))
    _config["system"]["device"] = "cpu"
    _config["data"]["num_classes"] = NUM_CLASSES
    _config["model"]["out_channels"] = NUM_CLASSES

    _model = build_unet3d(_config)

    if CHECKPOINT.exists():
        ckpt = torch.load(str(CHECKPOINT), map_location="cpu")
        _model.load_state_dict(ckpt["model_state_dict"])
        print(f"✓ Checkpoint loaded from {CHECKPOINT}")
    else:
        print("⚠  No checkpoint found — using random weights.")
        print("   Run: python main.py --config configs/config.yaml")

    _model.eval()
    return _model, _config


# ---------------------------------------------------------------------------
# NIfTI + inference
# ---------------------------------------------------------------------------

def _load_nifti(path) -> np.ndarray:
    import nibabel as nib
    arr = nib.load(str(path)).get_fdata(dtype=np.float32).squeeze()
    if arr.ndim != 3:
        raise ValueError(f"Expected 3-D volume, got {arr.shape}")
    return arr

def _infer(image_np: np.ndarray, config: dict) -> tuple[np.ndarray, np.ndarray]:
    crop = tuple(config.get("preprocessing", {}).get("crop_size", [80, 80, 80]))
    pipeline = PreprocessingPipeline(crop_size=crop)
    img_t, _ = pipeline(image_np, np.zeros(image_np.shape, dtype=np.int64))
    image_cropped = img_t.squeeze(0).numpy()
    model, _ = _load_model()
    with torch.no_grad():
        logits = model(img_t.unsqueeze(0))
    pred = logits.argmax(dim=1).squeeze(0).numpy()
    return image_cropped, pred

def _crop_label(image_np, label_np, config):
    crop = tuple(config.get("preprocessing", {}).get("crop_size", [80, 80, 80]))
    pipeline = PreprocessingPipeline(crop_size=crop)
    label_np = np.clip(label_np.astype(np.int64), 0, NUM_CLASSES - 1)
    _, lbl_t = pipeline(image_np, label_np)
    return lbl_t.numpy()


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _slice_fig(image, pred, label=None, alpha=0.5) -> plt.Figure:
    D, H, W = image.shape
    mid = [D // 2, H // 2, W // 2]
    names   = ["Axial", "Coronal", "Sagittal"]
    img_sl  = [image[mid[0]], image[:, mid[1]], image[:, :, mid[2]]]
    pred_sl = [pred[mid[0]],  pred[:, mid[1]],  pred[:, :, mid[2]]]
    lbl_sl  = ([label[mid[0]], label[:, mid[1]], label[:, :, mid[2]]]
               if label is not None else None)

    n_rows = 3 if label is not None else 2
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 4.8 * n_rows))
    fig.patch.set_facecolor("#0d0d1a")
    row_labels = (["MRI Image", "Segmentation Prediction"] +
                  (["Ground Truth Label"] if label is not None else []))

    for col in range(3):
        for row in range(n_rows):
            ax = axes[row, col]
            ax.set_facecolor("#0d0d1a")
            ax.axis("off")
            ax.imshow(img_sl[col].T, cmap="gray", origin="lower", aspect="auto")
            if row == 1:
                ax.imshow(pred_sl[col].T, cmap=CMAP, vmin=0, vmax=NUM_CLASSES-1,
                          origin="lower", alpha=alpha, aspect="auto")
            elif row == 2 and lbl_sl is not None:
                ax.imshow(lbl_sl[col].T, cmap=CMAP, vmin=0, vmax=NUM_CLASSES-1,
                          origin="lower", alpha=alpha, aspect="auto")
            if row == 0:
                ax.set_title(names[col], color="white", fontsize=14,
                             fontweight="bold", pad=8)
            if col == 0:
                ax.set_ylabel(row_labels[row], color="#ccccff", fontsize=10,
                              rotation=90, labelpad=10)

    patches = [mpatches.Patch(color=CMAP(i), label=f"{i}: {TISSUE_LABELS[i]}")
               for i in range(NUM_CLASSES)]
    fig.legend(handles=patches, loc="lower center", ncol=4, fontsize=10,
               facecolor="#0d0d1a", labelcolor="white", framealpha=0.5,
               bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout()
    return fig


def _dice_fig(dice: dict) -> plt.Figure:
    names  = [TISSUE_LABELS[i] for i in range(NUM_CLASSES)]
    scores = [dice.get(f"class_{i}", 0.0) for i in range(NUM_CLASSES)]
    colors = [CMAP(i) for i in range(NUM_CLASSES)]
    mean   = dice.get("mean_dice", 0.0)

    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor("#0d0d1a")
    ax.set_facecolor("#0d0d1a")
    bars = ax.bar(names, scores, color=colors, edgecolor="white", linewidth=0.6)
    ax.axhline(y=mean, color="white", linestyle="--", linewidth=1.5,
               label=f"Mean Dice = {mean:.4f}")
    ax.axhline(y=0.75, color="#44ff88", linestyle=":", linewidth=1.2,
               label="Target (0.75)")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Dice Score", color="white", fontsize=11)
    ax.set_title("Per-Class Dice Scores", color="white", fontweight="bold", fontsize=13)
    ax.tick_params(colors="white")
    for sp in ax.spines.values():
        sp.set_color("#333")
    plt.xticks(rotation=22, ha="right", color="white", fontsize=9)
    for bar, s in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width()/2, s + 0.025,
                f"{s:.3f}", ha="center", va="bottom", color="white", fontsize=8)
    ax.legend(facecolor="#0d0d1a", labelcolor="white", framealpha=0.5, fontsize=10)
    plt.tight_layout()
    return fig


def _metrics_fig() -> plt.Figure:
    rows = [
        ("Mean Dice",                   "< 0.5", "0.6–0.7", "0.75–0.82", "> 0.85"),
        ("Per-class Dice (WM/GM)",      "< 0.6", "0.7",     "0.80",      "> 0.88"),
        ("Per-class Dice (Ventricles)", "< 0.5", "0.65",    "0.75",      "> 0.82"),
        ("IoU (Jaccard)",               "< 0.4", "0.5–0.6", "0.65",      "> 0.75"),
    ]
    cell_colors = [["#1e1e3a","#ff4d4d33","#ffaa0033","#44aa4433","#00ff8833"] for _ in rows]
    fig, ax = plt.subplots(figsize=(12, 3.2))
    fig.patch.set_facecolor("#0d0d1a")
    ax.set_facecolor("#0d0d1a")
    ax.axis("off")
    tbl = ax.table(cellText=rows,
                   colLabels=["Metric","Bad","Acceptable","Good","SOTA"],
                   cellLoc="center", loc="center", cellColours=cell_colors)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 2.4)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#444")
        cell.set_text_props(color="white")
        if r == 0:
            cell.set_facecolor("#3a3a7a")
            cell.set_text_props(color="white", fontweight="bold")
    ax.set_title("FeTA 2.4 — Target Segmentation Metrics",
                 color="white", fontweight="bold", fontsize=13, pad=14)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Gradio callbacks
# ---------------------------------------------------------------------------

def load_subject(subject_id):
    if not subject_id or str(subject_id).strip() == "":
        return "⚠ No subject selected."
    img_path = _get_image_path(subject_id)
    lbl_path = _get_label_path(subject_id)
    lines = []
    if img_path:
        lines.append(f"✓ Image : {img_path.name}")
        lines.append(f"   {img_path.relative_to(ROOT)}")
    else:
        lines.append(f"✗ Image not found for {subject_id}")
    if lbl_path:
        lines.append(f"✓ Label : {lbl_path.name}")
    else:
        lines.append("✗ Label (dseg) not found")
    return "\n".join(lines)


def run_segmentation(subject_id, alpha):
    if not subject_id or str(subject_id).strip() == "":
        return None, None, "⚠ No subject selected."

    image_path = _get_image_path(subject_id)
    label_path = _get_label_path(subject_id)

    if not image_path:
        return None, None, f"❌ No image file found for {subject_id}"

    try:
        _, config = _load_model()
    except Exception as e:
        return None, None, f"❌ Model error: {e}"

    try:
        image_np = _load_nifti(image_path)
    except Exception as e:
        return None, None, f"❌ Could not load image: {e}"

    try:
        image_cropped, pred_np = _infer(image_np, config)
    except Exception as e:
        return None, None, f"❌ Inference error: {e}"

    label_cropped = None
    dice_results  = None
    if label_path:
        try:
            lbl_np        = _load_nifti(label_path)
            label_cropped = _crop_label(image_np, lbl_np, config)
            dice_results  = compute_multiclass_dice(
                torch.from_numpy(pred_np).unsqueeze(0),
                torch.from_numpy(label_cropped).unsqueeze(0),
                num_classes=NUM_CLASSES,
            )
        except Exception:
            label_cropped = None
            dice_results  = None

    slice_fig = _slice_fig(image_cropped, pred_np, label_cropped, alpha=alpha)
    dice_fig  = _dice_fig(dice_results) if dice_results else None

    ckpt_tag = "✓ Trained model" if CHECKPOINT.exists() else "⚠ Random weights"
    if dice_results:
        lines = [
            f"Checkpoint : {ckpt_tag}",
            f"Volume     : {list(image_cropped.shape)}",
            f"Subject    : {image_path.name}",
            "",
            f"{'Class':<25} Dice",
            "─" * 36,
        ]
        for i in range(NUM_CLASSES):
            bar = "█" * int(dice_results[f"class_{i}"] * 20)
            lines.append(f"[{i}] {TISSUE_LABELS[i]:<20}  {dice_results[f'class_{i}']:.4f}  {bar}")
        lines += ["─" * 36, f"Mean Dice : {dice_results['mean_dice']:.4f}"]
        status = "\n".join(lines)
    else:
        status = (
            f"Checkpoint : {ckpt_tag}\n"
            f"Volume     : {list(image_cropped.shape)}\n"
            f"Subject    : {image_path.name}\n\n"
            "✓ Segmentation done.\n\n"
            "No dseg label file found —\nDice scores unavailable."
        )

    return slice_fig, dice_fig, status


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

CSS = """
.gradio-container { max-width: 1350px !important; margin: auto; }
h1 { text-align:center; font-size: 2rem; margin-bottom: 2px; }
.subtitle { text-align:center; color:#9999cc; font-size:1rem; margin-bottom:20px; }
"""

def build_app():
    ckpt_exists     = CHECKPOINT.exists()
    n_subjects      = len(SUBJECT_LIST)
    default_subject = SUBJECT_LIST[0] if SUBJECT_LIST else None
    default_status  = load_subject(default_subject) if default_subject else "⚠ No subjects found."

    print(f"\n{'='*55}")
    print(f"  BIDS root  : {BIDS_ROOT}")
    print(f"  Labels root: {LABELS_ROOT}")
    print(f"  Subjects   : {n_subjects}")
    if n_subjects:
        print(f"  List       : {SUBJECT_LIST[:8]}{'...' if n_subjects > 8 else ''}")
    print(f"{'='*55}\n")

    with gr.Blocks(
        title="Fetal Brain MRI Segmentation",
        theme=gr.themes.Base(primary_hue="violet", neutral_hue="slate"),
        css=CSS,
    ) as demo:

        gr.Markdown("# 🧠 Fetal Brain MRI Segmentation")
        gr.HTML(
            "<p class='subtitle'>Domain-Generalizable Multi-Tissue Segmentation · "
            "3D Attention U-Net · FeTA 2.4 Dataset</p>"
        )

        with gr.Tab("🔬 Segmentation"):
            with gr.Row():
                with gr.Column(scale=1, min_width=320):

                    gr.Markdown("### Step 1 — Pick a Subject")
                    if n_subjects == 0:
                        gr.Markdown(
                            "⚠ **No subjects found.**\n\n"
                            f"BIDS root searched: `{BIDS_ROOT}`\n\n"
                            "Check terminal output for the full diagnostic."
                        )
                        subject_dd = gr.Dropdown(choices=[], label="Subject", interactive=False)
                    else:
                        subject_dd = gr.Dropdown(
                            choices=SUBJECT_LIST,
                            value=default_subject,
                            label=f"Subject ({n_subjects} available)",
                            interactive=True,
                        )

                    load_status = gr.Textbox(
                        label="Load status", lines=4,
                        interactive=False, value=default_status,
                    )

                    gr.Markdown("### Step 2 — Run")
                    alpha_slider = gr.Slider(
                        minimum=0.1, maximum=0.9, value=0.5, step=0.05,
                        label="Overlay opacity",
                    )
                    run_btn = gr.Button("▶  Run Segmentation", variant="primary", size="lg")

                    gr.Markdown("### Results")
                    status_box = gr.Textbox(
                        label="Dice Scores", lines=16, interactive=False,
                        placeholder="Results will appear here after segmentation...",
                    )

                with gr.Column(scale=2):
                    gr.Markdown(
                        f"**Checkpoint:** {'✓ Trained model loaded' if ckpt_exists else '⚠ No checkpoint — random predictions. Run `python main.py` to train.'}"
                    )
                    slice_plot = gr.Plot(label="Axial · Coronal · Sagittal")
                    dice_plot  = gr.Plot(label="Per-Class Dice Scores")

            subject_dd.change(
                fn=load_subject,
                inputs=[subject_dd],
                outputs=[load_status],
            )
            run_btn.click(
                fn=run_segmentation,
                inputs=[subject_dd, alpha_slider],
                outputs=[slice_plot, dice_plot, status_box],
            )

        with gr.Tab("📊 Target Metrics"):
            gr.Markdown(
                "### FeTA 2.4 Benchmark Scores\n"
                "**Mean Dice > 0.75** is a strong result. "
                "The hardest classes are External CSF and Deep Gray Matter."
            )
            metrics_btn  = gr.Button("Show Metrics Table")
            metrics_plot = gr.Plot()
            metrics_btn.click(fn=_metrics_fig, outputs=metrics_plot)

            gr.Markdown("""
            ### Domain Shift Experiment
            | Experiment | Train | Test | Expected Dice |
            |---|---|---|---|
            | In-domain | Site A | Site A | ~0.80+ |
            | Cross-domain (no adaptation) | Site A | Site B | ~0.60–0.70 |
            | With domain generalisation | Site A | Site B | ~0.75+ |
            """)

        with gr.Tab("ℹ️ About"):
            gr.Markdown(f"""
            ### Model Architecture
            - **3D Attention U-Net** with skip connections + attention gates
            - Input: `[1, 80, 80, 80]` — single-channel T2-weighted MRI
            - Output: `[8, 80, 80, 80]` — 8-class segmentation logits

            ### Tissue Classes
            | Label | Tissue | Label | Tissue |
            |---|---|---|---|
            | 0 | Background | 4 | Ventricles |
            | 1 | External CSF | 5 | Cerebellum |
            | 2 | Gray Matter | 6 | Deep Gray Matter |
            | 3 | White Matter | 7 | Brainstem |

            ### Detected Paths
            - **BIDS root:** `{BIDS_ROOT}`
            - **Labels root:** `{LABELS_ROOT}`
            - **Checkpoint:** `{"✓ Found" if ckpt_exists else "⚠ Not found"}`
            """)

    return demo


if __name__ == "__main__":
    print("Loading model...")
    _load_model()
    print(f"✓ Found {len(SUBJECT_LIST)} subjects")
    print("✓ Starting app — open http://127.0.0.1:7860")
    build_app().launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
    )