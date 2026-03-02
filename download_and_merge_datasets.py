"""
AcciVision — Dataset Downloader & Merger
=========================================
Downloads free public vehicle-accident datasets from Roboflow Universe,
remaps their class IDs to match AcciVision's 3-class scheme
  ( 0 = car  |  1 = accident  |  2 = truck )
and merges everything into the AcciiVision directory that train.py uses.

Prerequisites
-------------
1.  Create a FREE Roboflow account at https://roboflow.com
2.  Get your API key from: https://app.roboflow.com/settings/api
3.  Either:
      a) Run:  python download_and_merge_datasets.py --api-key YOUR_KEY
      b) Set env var:  ROBOFLOW_API_KEY=YOUR_KEY  then run without --api-key

After this script finishes, run train.py to retrain with the merged dataset.
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path

import requests

# ── Destination (AcciiVision) layout ────────────────────────────────────────
DEST_ROOT   = Path("AcciiVision")
DEST_DIRS   = {
    "train_img": DEST_ROOT / "images"  / "training",
    "train_lbl": DEST_ROOT / "labels"  / "training",
    "val_img":   DEST_ROOT / "images"  / "validation",
    "val_lbl":   DEST_ROOT / "labels"  / "validation",
}

# ── AcciVision class scheme  (canonical) ────────────────────────────────────
TARGET_CLASSES = {"car": 0, "accident": 1, "truck": 2}

# Keywords that map a source class name → one of our canonical names.
# Matching is case-insensitive; first matching keyword wins.
CLASS_KEYWORD_MAP = {
    "accident":  ["accident", "crash", "collision", "wreck",
                  "accidente", "_accident", "severe", "moderate",
                  "mild", "minor", "rollover", "collid"],
    "car":       ["car", "vehicle", "auto", "sedan", "suv",
                  "hatchback", "coupe", "motorbike", "motorcycle",
                  "motor cycle", "bike", "van", "bus", "ambulance"],
    "truck":     ["truck", "lorry", "pickup", "semi", "trailer",
                  "heavy"],
}  # Note: "non-accident", "noaccident", "normal" etc. will return None (skip)

# ── Free public Roboflow datasets to download ────────────────────────────────
# Format:  (workspace, project, version)
# All are public CC-BY datasets on Roboflow Universe.
# Verified working via: https://api.roboflow.com/{workspace}/{project}?api_key=KEY
ROBOFLOW_DATASETS = [
    # 1 — 200 imgs / 17 MB: car + Accident + mild/moderate/severe labels
    ("self-ixih1",                   "accident-detection-qgglm",    1),
    # 2 — 3.1k imgs / 185 MB: accident / non-accident binary labels
    ("accident-detection-lu7np",     "accident-detection-tqpa0",    2),
    # 3 — 4k imgs / 191 MB: single 'accident' class dashcam footage
    ("accident-detection-d4mcs",     "accident-detection-djvd3",    1),
    # 4 — 9.5k imgs / 385 MB: accident-only class
    ("tan-cvesi",                    "accident-detection-pqe7n",    9),
    # 5 — 11.7k imgs / 945 MB: severe/moderate crash scenes
    ("accident-detection-ffdrf",     "accident-detection-8dvh5",    1),
]

# ── Temporary download root ──────────────────────────────────────────────────
DOWNLOAD_ROOT = Path("_dataset_downloads")


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def ensure_dirs():
    for d in DEST_DIRS.values():
        d.mkdir(parents=True, exist_ok=True)


def file_hash_md5(path: Path, chunk=65536) -> str:
    """Quick MD5 fingerprint so we can skip duplicate images."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# Negative/background class names — annotated boxes we must SKIP entirely.
# Checked BEFORE the positive keyword map to prevent false matches
# (e.g. "non-accident" contains "accident").
_NEGATIVE_CLASS_PATTERNS = [
    "no_accident", "non-accident", "non_accident", "noaccident",
    "no accident", "normal", "no_crash", "no crash", "non_crash",
    "background", "unlabeled", "unlabelled",
]


def remap_class_name(raw_name: str) -> int | None:
    """Map an arbitrary class name string → our class id, or None to discard."""
    n = raw_name.strip().lower()
    # Reject negative/background labels first so "non-accident" is not caught
    # by the positive "accident" keyword below.
    for neg in _NEGATIVE_CLASS_PATTERNS:
        if neg in n:
            return None
    for canonical, keywords in CLASS_KEYWORD_MAP.items():
        for kw in keywords:
            if kw in n:
                return TARGET_CLASSES[canonical]
    return None  # unknown class — drop this annotation


def remap_label_file(src_label: Path, src_classes: list[str]) -> list[str] | None:
    """
    Re-read a YOLO label file and remap every class id.
    Returns a list of remapped lines (may be empty if all annots are dropped).
    Returns None if the source file doesn't exist.
    """
    if not src_label.exists():
        return None
    out_lines = []
    with open(src_label) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            src_cls_id = int(parts[0])
            if src_cls_id >= len(src_classes):
                continue
            src_name = src_classes[src_cls_id]
            new_id = remap_class_name(src_name)
            if new_id is None:
                continue
            out_lines.append(f"{new_id} {' '.join(parts[1:])}")
    return out_lines


def copy_to_dest(img_src: Path, lbl_lines: list[str],
                 img_dir: Path, lbl_dir: Path,
                 seen_hashes: set, prefix: str) -> bool:
    """
    Copy an image + its remapped labels into the destination directories.
    Returns True if the image was actually copied (not a duplicate).
    """
    if not img_src.exists():
        return False

    h = file_hash_md5(img_src)
    if h in seen_hashes:
        return False          # exact duplicate — skip
    seen_hashes.add(h)

    stem = f"{prefix}_{img_src.stem}"
    ext  = img_src.suffix.lower() or ".jpg"

    dst_img = img_dir / f"{stem}{ext}"
    dst_lbl = lbl_dir / f"{stem}.txt"

    shutil.copy2(img_src, dst_img)
    with open(dst_lbl, "w") as f:
        f.write("\n".join(lbl_lines) + ("\n" if lbl_lines else ""))
    return True


def load_classes_txt(classes_file: Path) -> list[str]:
    if not classes_file.exists():
        return []
    with open(classes_file) as f:
        return [line.strip() for line in f if line.strip()]


def find_classes(dataset_dir: Path) -> list[str]:
    """
    Try to locate a classes.txt or data.yaml inside the downloaded dataset
    and extract the class list.
    """
    # Check for data.yaml
    for yaml_path in dataset_dir.rglob("data.yaml"):
        try:
            import yaml
            with open(yaml_path) as f:
                data = yaml.safe_load(f)
            if "names" in data:
                names = data["names"]
                if isinstance(names, dict):
                    return [names[i] for i in sorted(names.keys())]
                if isinstance(names, list):
                    return names
        except Exception:
            pass

    # Fallback: classes.txt
    for cls_path in dataset_dir.rglob("classes.txt"):
        classes = load_classes_txt(cls_path)
        if classes:
            return classes

    # Fallback: _classes.csv (Roboflow sometimes uses this)
    for csv_path in dataset_dir.rglob("_classes.csv"):
        try:
            import csv
            with open(csv_path) as f:
                reader = csv.reader(f)
                for row in reader:
                    return [c.strip() for c in row if c.strip()]
        except Exception:
            pass

    return []


def ingest_yolo_split(images_dir: Path, labels_dir: Path,
                      src_classes: list[str],
                      dest_img_dir: Path, dest_lbl_dir: Path,
                      seen_hashes: set, prefix: str) -> int:
    """
    Walk one images/ + labels/ split directory and copy everything
    (with class remapping) into the destination.
    Returns the number of images actually added.
    """
    added = 0
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for img_path in sorted(images_dir.iterdir()):
        if img_path.suffix.lower() not in img_exts:
            continue

        lbl_path = labels_dir / (img_path.stem + ".txt")
        lbl_lines = remap_label_file(lbl_path, src_classes)
        if lbl_lines is None:
            # No label file at all — create an empty one (negative sample)
            lbl_lines = []

        if copy_to_dest(img_path, lbl_lines,
                        dest_img_dir, dest_lbl_dir,
                        seen_hashes, prefix):
            added += 1

    return added


def ingest_dataset(dataset_dir: Path, prefix: str, seen_hashes: set) -> tuple[int, int]:
    """
    Auto-detect the layout of a downloaded Roboflow dataset and ingest it.
    Returns (train_added, val_added).
    """
    src_classes = find_classes(dataset_dir)
    if not src_classes:
        print(f"    ⚠  Could not find class list in {dataset_dir} — skipping.")
        return 0, 0

    print(f"    Source classes: {src_classes}")

    train_added = val_added = 0

    # Try standard Roboflow layout: train/images + train/labels, etc.
    for split_name, dest_img, dest_lbl in [
        ("train", DEST_DIRS["train_img"], DEST_DIRS["train_lbl"]),
        ("valid", DEST_DIRS["val_img"],   DEST_DIRS["val_lbl"]),
        ("test",  DEST_DIRS["train_img"], DEST_DIRS["train_lbl"]),  # test → train
    ]:
        split_img_dir = dataset_dir / split_name / "images"
        split_lbl_dir = dataset_dir / split_name / "labels"

        if split_img_dir.exists() and split_lbl_dir.exists():
            n = ingest_yolo_split(split_img_dir, split_lbl_dir,
                                  src_classes, dest_img, dest_lbl,
                                  seen_hashes, f"{prefix}_{split_name}")
            if split_name in ("train", "test"):
                train_added += n
            else:
                val_added += n

    # Some datasets are flat: images/ + labels/ at root
    if train_added == 0 and val_added == 0:
        flat_img = dataset_dir / "images"
        flat_lbl = dataset_dir / "labels"
        if flat_img.exists() and flat_lbl.exists():
            n = ingest_yolo_split(flat_img, flat_lbl,
                                  src_classes,
                                  DEST_DIRS["train_img"], DEST_DIRS["train_lbl"],
                                  seen_hashes, prefix)
            train_added += n

    return train_added, val_added


# ─────────────────────────────────────────────────────────────────────────────
#  Roboflow download
# ─────────────────────────────────────────────────────────────────────────────

def download_roboflow_dataset(api_key: str,
                               workspace: str, project: str, version: int,
                               dest_dir: Path) -> Path | None:
    """
    Download one Roboflow dataset in YOLOv8 format using the REST API directly.
    Returns the path to the unzipped dataset folder, or None on failure.
    """
    import urllib.request
    out_dir = dest_dir / f"{workspace}_{project}_v{version}"
    if out_dir.exists() and any(out_dir.iterdir()):
        print(f"    Already downloaded {workspace}/{project} v{version}, reusing.")
        return out_dir

    try:
        # Step 1: get the export download link
        info_url = (f"https://api.roboflow.com/{workspace}/{project}/{version}"
                    f"/yolov8?api_key={api_key}")
        req = urllib.request.Request(info_url)
        resp = urllib.request.urlopen(req, timeout=30)
        data = json.loads(resp.read())
        link = data.get("export", {}).get("link")
        if not link:
            print(f"    ✗ No download link in API response for {workspace}/{project}")
            return None
        size_mb = data.get("export", {}).get("size", 0)
        print(f"    Downloading {workspace}/{project} v{version} ({size_mb:.0f} MB) …")

        # Step 2: download & unzip
        zip_path = dest_dir / f"{workspace}_{project}_v{version}.zip"
        resp2 = requests.get(link, stream=True, timeout=300,
                             headers={"User-Agent": "AcciVision/1.0"})
        resp2.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in resp2.iter_content(131072):
                f.write(chunk)

        out_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(out_dir)
        zip_path.unlink(missing_ok=True)
        return out_dir
    except Exception as e:
        print(f"    ✗ Failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  Fallback: download from direct ZIP URLs (no API key needed)
# ─────────────────────────────────────────────────────────────────────────────
# Publicly hosted accident/vehicle labelled datasets (CC-BY / public domain).

# ─────────────────────────────────────────────────────────────────────────────
#  Direct download helpers                (no API key needed)
# ─────────────────────────────────────────────────────────────────────────────

def download_direct(info: dict, dest_dir: Path) -> Path | None:
    """Download a ZIP dataset from a direct URL and unzip it."""
    name = info["name"]
    url  = info["url"]
    zip_path = dest_dir / f"{name}.zip"
    out_dir  = dest_dir / name

    if out_dir.exists():
        print(f"    Already downloaded {name}, reusing.")
        return out_dir

    print(f"    Downloading {name} …")
    try:
        resp = requests.get(url, stream=True, timeout=90,
                            headers={"User-Agent": "AcciVision/1.0"},
                            allow_redirects=True)
        resp.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in resp.iter_content(65536):
                f.write(chunk)
        out_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(out_dir)
        zip_path.unlink(missing_ok=True)
        return out_dir
    except Exception as e:
        print(f"    ✗ Failed: {e}")
        zip_path.unlink(missing_ok=True)
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  COCO128 vehicle extraction  (no API key, auto-downloaded by ultralytics)
# ─────────────────────────────────────────────────────────────────────────────
# COCO class IDs we keep  →  our class id
# 2=car→0  7=truck→2  5=bus→2  3=motorcycle→0
COCO_ID_TO_OURS = {2: 0, 7: 2, 5: 2, 3: 0}


def ingest_coco128(seen_hashes: set) -> tuple[int, int]:
    """
    Download COCO128 (128-image COCO sample bundled with ultralytics),
    extract only vehicle-class annotations, remap to our IDs, and add to
    AcciiVision.  Returns (added_train, added_val).
    """
    print("\n[COCO128] Downloading COCO128 via ultralytics (free, no key) …")
    coco128_dir = DOWNLOAD_ROOT / "coco128"
    img_dir     = coco128_dir / "images"  / "train2017"
    lbl_dir     = coco128_dir / "labels"  / "train2017"

    if not img_dir.exists():
        try:
            from ultralytics.data.utils import download as ul_download
            ul_download(
                "https://ultralytics.com/assets/coco128.zip",
                dir=DOWNLOAD_ROOT, unzip=True, delete=True, threads=1,
            )
        except Exception as e:
            print(f"    ✗ COCO128 download failed: {e}")
            return 0, 0

    if not img_dir.exists() or not lbl_dir.exists():
        print("    ✗ COCO128 directories missing after download.")
        return 0, 0

    added = 0
    for img_path in sorted(img_dir.glob("*.jpg")):
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue
        remapped = []
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                coco_cls = int(parts[0])
                if coco_cls in COCO_ID_TO_OURS:
                    remapped.append(
                        f"{COCO_ID_TO_OURS[coco_cls]} {' '.join(parts[1:])}")
        if not remapped:
            continue
        if copy_to_dest(img_path, remapped,
                        DEST_DIRS["train_img"], DEST_DIRS["train_lbl"],
                        seen_hashes, "coco128"):
            added += 1

    print(f"    ✓  COCO128 vehicle images added: {added}")
    return added, 0


# ─────────────────────────────────────────────────────────────────────────────
#  Roboflow Universe public export prober  (no key for public datasets)
# ─────────────────────────────────────────────────────────────────────────────
# These URLs use Roboflow Universe's unauthenticated public export endpoint.
# Only public (Community) datasets work here.
ROBOFLOW_PUBLIC_EXPORTS = [
    {
        "name":    "rf_pub_accident_det",
        "url":     "https://universe.roboflow.com/ds/"
                   "b2ydlTEUJl?key=kzd12345/yolov8",   # placeholder — see NOTE
        "classes": ["accident"],
    },
]
# NOTE: Roboflow Universe public export links have the form
#   https://universe.roboflow.com/ds/<dataset_token>?key=<api_key>
# You can find them by clicking "Download Dataset" on any public project page.
# Paste a download link above and the script will fetch it automatically.


# ─────────────────────────────────────────────────────────────────────────────
#  Hash existing images so we don't duplicate them
# ─────────────────────────────────────────────────────────────────────────────

def hash_existing_images() -> set[str]:
    seen = set()
    for d in (DEST_DIRS["train_img"], DEST_DIRS["val_img"]):
        if d.exists():
            for p in d.iterdir():
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                    seen.add(file_hash_md5(p))
    print(f"  Existing images fingerprinted: {len(seen)}")
    return seen


# ─────────────────────────────────────────────────────────────────────────────
#  Update data.yaml with final counts
# ─────────────────────────────────────────────────────────────────────────────

def update_data_yaml():
    n_train = len(list(DEST_DIRS["train_img"].glob("*.jpg"))) + \
              len(list(DEST_DIRS["train_img"].glob("*.jpeg"))) + \
              len(list(DEST_DIRS["train_img"].glob("*.png")))
    n_val   = len(list(DEST_DIRS["val_img"].glob("*.jpg"))) + \
              len(list(DEST_DIRS["val_img"].glob("*.jpeg"))) + \
              len(list(DEST_DIRS["val_img"].glob("*.png")))

    yaml_content = f"""# AcciVision merged dataset — auto-generated by download_and_merge_datasets.py
# Training images : {n_train}
# Validation images: {n_val}

path: AcciiVision

train: images/training
val:   images/validation

nc: 3
names:
  0: car
  1: accident
  2: truck
"""
    Path("data.yaml").write_text(yaml_content)
    print(f"\n  data.yaml updated  →  train: {n_train}  |  val: {n_val}")


# ─────────────────────────────────────────────────────────────────────────────
#  Update coco1.txt
# ─────────────────────────────────────────────────────────────────────────────

def update_coco1_txt():
    Path("coco1.txt").write_text("car\naccident\ntruck\n")
    print("  coco1.txt updated  →  car / accident / truck")


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download free accident datasets and merge into AcciiVision.")
    parser.add_argument(
        "--api-key", default=os.environ.get("ROBOFLOW_API_KEY", ""),
        help="Roboflow API key (or set env var ROBOFLOW_API_KEY). "
             "Get a free key at https://app.roboflow.com/settings/api")
    parser.add_argument(
        "--skip-roboflow", action="store_true",
        help="Skip Roboflow downloads (use only direct-URL datasets).")
    parser.add_argument(
        "--skip-direct", action="store_true",
        help="Skip direct-URL downloads (use only Roboflow datasets).")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  AcciVision — Dataset Downloader & Merger")
    print("="*60)

    # Prepare destination directories
    ensure_dirs()

    # Fingerprint what already exists
    seen_hashes = hash_existing_images()

    total_train = total_val = 0
    DOWNLOAD_ROOT.mkdir(exist_ok=True)

    # ── 1. Roboflow datasets ──────────────────────────────────────────────
    if not args.skip_roboflow:
        if not args.api_key:
            print("\n[Roboflow] No API key provided.")
            print("  → Get a FREE key at https://app.roboflow.com/settings/api")
            print("  → Re-run:  python download_and_merge_datasets.py --api-key YOUR_KEY")
            print("  → Skipping Roboflow downloads.\n")
        else:
            for workspace, project, version in ROBOFLOW_DATASETS:
                print(f"\n[Roboflow] {workspace}/{project} v{version}")
                dl_path = download_roboflow_dataset(
                    args.api_key, workspace, project, version, DOWNLOAD_ROOT)

                if dl_path:
                    prefix = f"rf_{project.replace('-','_')}"
                    t, v = ingest_dataset(dl_path, prefix, seen_hashes)
                    print(f"    ✓  Added  train={t}  val={v}")
                    total_train += t
                    total_val   += v

    # ── 2. COCO128 free vehicle data (no key) ────────────────────────────
    if not args.skip_direct:
        ct, _ = ingest_coco128(seen_hashes)
        total_train += ct

    # ── 3. Roboflow Universe public export links (no key) ────────────────
    if not args.skip_direct:
        for info in ROBOFLOW_PUBLIC_EXPORTS:
            url = info["url"]
            # Skip placeholder entries (contain 'placeholder' or dummy tokens)
            if "placeholder" in url or "kzd12345" in url:
                continue
            print(f"\n[Public Export] {info['name']}")
            dl_path = download_direct(info, DOWNLOAD_ROOT)
            if dl_path:
                if "classes" in info:
                    (dl_path / "classes.txt").write_text("\n".join(info["classes"]))
                prefix = f"pub_{info['name']}"
                t, v = ingest_dataset(dl_path, prefix, seen_hashes)
                print(f"    ✓  Added  train={t}  val={v}")
                total_train += t
                total_val   += v

    # ── 4. Finalise ───────────────────────────────────────────────────────
    print("\n" + "-"*60)
    print(f"  New images added  →  train: {total_train}  |  val: {total_val}")

    update_data_yaml()
    update_coco1_txt()

    # Final counts
    n_tr = sum(1 for p in DEST_DIRS["train_img"].iterdir()
               if p.suffix.lower() in {".jpg",".jpeg",".png"})
    n_va = sum(1 for p in DEST_DIRS["val_img"].iterdir()
               if p.suffix.lower() in {".jpg",".jpeg",".png"})

    print(f"\n  Dataset totals after merge:")
    print(f"    Training   : {n_tr} images")
    print(f"    Validation : {n_va} images")
    print(f"    Total      : {n_tr + n_va} images")
    print("\n  Run 'python train.py' to retrain with the merged dataset.\n")


if __name__ == "__main__":
    main()
