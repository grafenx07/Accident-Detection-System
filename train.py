"""
AcciVision — YOLOv8 Training Script
====================================
Fine-tunes a YOLOv8 model on the AcciiVision accident dataset using
best-practice settings to maximise real-world accuracy and minimise
false positives (hallucinations).

Usage
-----
    python train.py                    # default: yolov8n, 50 epochs (CPU-friendly)
    python train.py --model yolov8s    # slightly larger model
    python train.py --epochs 100       # more epochs if you have time
    python train.py --no-freeze        # train all layers (slow on CPU)
    python train.py --resume           # resume from last checkpoint

After training the best weights are auto-copied to best.pt in the project root.
"""

import argparse
import os
from pathlib import Path

from ultralytics import YOLO


# ── Hyper-parameters ──────────────────────────────────────────────────────────
# yolov8n = nano (3M params) — best for CPU training, ~30s/epoch
# yolov8s = small (11M params) — better accuracy, ~90s/epoch on CPU
# yolov8m = medium (25M params) — GPU recommended, ~75 min/epoch on CPU
DEFAULT_MODEL  = "yolov8n.pt"
DATA_YAML      = "data.yaml"
# Use absolute path to avoid ultralytics double-nesting the save directory
PROJECT_DIR    = str(Path(__file__).resolve().parent / "runs")
RUN_NAME       = "accivision_v1"
IMG_SIZE       = 416             # 416 px is ~2x faster than 640 on CPU with minimal accuracy loss
EPOCHS         = 50              # 50 epochs is sufficient with a pre-trained backbone
PATIENCE       = 15              # early-stopping patience
# Freeze the backbone (first 10 layers) — only fine-tune neck + detection head.
# This is 5-10x faster and enough for domain adaptation on a pre-trained model.
FREEZE_LAYERS  = 10              # set to 0 to train all layers
BATCH          = 4               # safe for CPU; increase to 8/16 if you have a GPU
WORKERS        = 0               # 0 = main thread only (required on Windows)
DEVICE         = ""              # "" = auto-detect (GPU if available, else CPU)

# ── Augmentation settings (lightweight for CPU — no mixup/copy_paste) ────────
AUG = dict(
    # Geometric
    degrees     = 5.0,    # ±5° rotation  — handles tilted cameras
    translate   = 0.1,    # ±10 % translation
    scale       = 0.5,    # ±50 % scale jitter  — handles varied distances
    shear       = 2.0,    # ±2° shear
    perspective = 0.0005, # slight perspective warp
    flipud      = 0.0,    # no vertical flip (cars don't fly)
    fliplr      = 0.5,    # 50 % horizontal mirror

    # Colour / photometric
    hsv_h       = 0.015,  # hue jitter       — varies lighting colour
    hsv_s       = 0.7,    # saturation jitter — handles over/under-exposed cams
    hsv_v       = 0.4,    # value jitter      — handles dark / bright scenes

    # Mosaic augmentation (CPU-safe)
    mosaic      = 1.0,    # always-on mosaic  — forces learning from context
    mixup       = 0.0,    # disabled — expensive on CPU
    copy_paste  = 0.0,    # disabled — expensive on CPU

    # Close-mosaic: disable mosaic for the final N epochs for fine-tuning
    close_mosaic = 10,
)

# ── Loss weights (slightly boost classification + object confidence losses) ──
LOSS = dict(
    box  = 7.5,   # bounding-box regression loss weight (default 7.5)
    cls  = 0.8,   # classification loss weight (default 0.5) — penalise wrong class
    dfl  = 1.5,   # distribution focal loss weight (default 1.5)
)

# ── NMS / inference thresholds written into the saved model's config ─────────
CONF_THRESHOLD = 0.55   # minimum detection confidence for NMS during val
IOU_THRESHOLD  = 0.45   # NMS IoU threshold  — lower → fewer duplicate boxes
MAX_DET        = 50     # max detections per image


def parse_args():
    p = argparse.ArgumentParser(description="AcciVision YOLOv8 training")
    p.add_argument("--model",     default=DEFAULT_MODEL,
                   help="Pretrained weights (e.g. yolov8n.pt / yolov8s.pt / yolov8m.pt)")
    p.add_argument("--epochs",    type=int, default=EPOCHS)
    p.add_argument("--batch",     type=int, default=BATCH)
    p.add_argument("--imgsz",     type=int, default=IMG_SIZE)
    p.add_argument("--device",    default=DEVICE,
                   help="cuda device (0, 1, …) or 'cpu'. Empty = auto.")
    p.add_argument("--resume",    action="store_true",
                   help="Resume training from the last checkpoint")
    p.add_argument("--name",      default=RUN_NAME,
                   help="Experiment run name")
    p.add_argument("--no-freeze", action="store_true",
                   help="Train all layers (default: freeze backbone for faster CPU training)")
    return p.parse_args()


def train(args):
    data_path = Path(DATA_YAML).resolve()
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset config not found: {data_path}\n"
            "Make sure data.yaml is present and the AcciiVision folder is set up correctly."
        )

    print(f"\n{'='*60}")
    print(f"  AcciVision — YOLOv8 Training")
    print(f"  Model   : {args.model}")
    print(f"  Dataset : {data_path}")
    print(f"  Epochs  : {args.epochs}  |  Batch: {args.batch}  |  Img: {args.imgsz}px")
    print(f"{'='*60}\n")

    # Load model (downloads pretrained COCO weights on first run)
    model = YOLO(args.model)

    freeze = None if args.no_freeze else FREEZE_LAYERS
    if freeze:
        print(f"  Backbone freeze : first {freeze} layers (use --no-freeze to train all)")
    else:
        print("  Backbone freeze : disabled (training all layers)")

    # ── Training ──────────────────────────────────────────────────────────────
    results = model.train(
        data      = str(data_path),
        epochs    = args.epochs,
        patience  = PATIENCE,
        batch     = args.batch,
        imgsz     = args.imgsz,
        device    = args.device if args.device else None,
        workers   = WORKERS,
        project   = PROJECT_DIR,
        name      = args.name,
        resume    = args.resume,
        exist_ok  = True,
        freeze    = freeze,         # freeze backbone for faster CPU transfer-learning

        # Optimiser
        optimizer = "AdamW",
        lr0       = 0.001,
        lrf       = 0.01,
        momentum  = 0.937,
        weight_decay = 0.0005,
        warmup_epochs   = 3.0,
        warmup_momentum = 0.8,
        warmup_bias_lr  = 0.1,
        cos_lr    = True,

        # NMS thresholds used during validation
        conf      = CONF_THRESHOLD,
        iou       = IOU_THRESHOLD,
        max_det   = MAX_DET,

        # Loss weights
        box       = LOSS["box"],
        cls       = LOSS["cls"],
        dfl       = LOSS["dfl"],

        # Augmentation
        **AUG,

        # Output & diagnostics
        save        = True,
        save_period = 10,
        plots       = True,
        verbose     = True,
        amp         = True,

        # Overlap mask
        overlap_mask = True,
    )

    best_weights = Path(PROJECT_DIR) / args.name / "weights" / "best.pt"  # absolute path already
    print(f"\nTraining complete.")
    print(f"Best weights : {best_weights.resolve()}")

    if best_weights.exists():
        import shutil
        dest = Path("best.pt")
        shutil.copy(best_weights, dest)
        print(f"Copied best.pt → {dest.resolve()}")
    else:
        print("WARNING: best.pt not found — check run directory for errors.")

    return results


def validate(model_path: str = "best.pt"):
    """Quick validation pass on the val set to confirm mAP after training."""
    data_path = Path(DATA_YAML).resolve()
    model = YOLO(model_path)
    metrics = model.val(
        data   = str(data_path),
        imgsz  = IMG_SIZE,
        conf   = CONF_THRESHOLD,
        iou    = IOU_THRESHOLD,
        max_det = MAX_DET,
        verbose = True,
    )
    print(f"\nValidation mAP50      : {metrics.box.map50:.4f}")
    print(f"Validation mAP50-95   : {metrics.box.map:.4f}")
    print(f"Precision             : {metrics.box.mp:.4f}")
    print(f"Recall                : {metrics.box.mr:.4f}")
    return metrics


if __name__ == "__main__":
    args = parse_args()
    train(args)
    # Validation metrics are already printed by YOLO at the end of training.
    # To run a standalone validation after training:
    #   python train.py --validate-only   (or use: yolo val model=best.pt data=data.yaml)
