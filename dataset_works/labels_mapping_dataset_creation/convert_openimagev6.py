import os
import shutil
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# === PATHS ===
DATA_ROOT        = Path("/media/sameerhashmi/ran_epav_disk/Sameer_dataset_from_smb/data_sameer/open-images-v6")
OUTPUT_ROOT      = Path("/media/sameerhashmi/ran_epav_disk/Sameer_dataset_from_smb/converted_datasets/open-images-v6")
MASTER_NAMES_PTH = Path("./master.names")

SPLITS = ["train", "validation", "test"]

def load_master_names(pth):
    names = [x.strip() for x in pth.read_text().splitlines() if x.strip()]
    return {n: i for i, n in enumerate(names)}

def normalize(name: str) -> str:
    return name.lower().strip().replace(" ", "_").replace("/", "_")

# Load unified name→index map
name2idx = load_master_names(MASTER_NAMES_PTH)

def convert_split(split: str):
    print(f"\n→ Converting {split} (fast mode)")
    split_dir   = DATA_ROOT / split
    img_src_dir = split_dir / "data"
    lbl_csv     = split_dir / "labels" / "detections.csv"
    classes_csv = split_dir / "metadata" / "classes.csv"

    # Skip if directory missing
    if not img_src_dir.exists():
        print(f"Warning: {img_src_dir} does not exist, skipping split.")
        return

    # Pre-build a map: image_id → full Path (avoids repeated globbing)
    id2path = {}
    for p in img_src_dir.iterdir():
        if p.is_file():
            img_id = p.name.split(".", 1)[0]
            id2path[img_id] = p

    # Read classes.csv to map LabelName → normalized DisplayName
    cls_df = pd.read_csv(classes_csv, header=None,
                         names=["LabelName","DisplayName"], dtype=str)
    mid2name = {
        row.LabelName: normalize(row.DisplayName)
        for _, row in cls_df.iterrows()
    }

    # Read only required columns (speeds up CSV parsing)
    df = pd.read_csv(
        lbl_csv,
        usecols=["ImageID","LabelName","XMin","XMax","YMin","YMax"],
        dtype={
            "ImageID": str,
            "LabelName": str,
            "XMin": float,
            "XMax": float,
            "YMin": float,
            "YMax": float,
        },
        low_memory=False
    )

    # Prepare output directories
    img_dst_dir = OUTPUT_ROOT / "images" / split
    lbl_dst_dir = OUTPUT_ROOT / "labels" / split
    img_dst_dir.mkdir(parents=True, exist_ok=True)
    lbl_dst_dir.mkdir(parents=True, exist_ok=True)

    # Process each image's detections
    for img_id, group in tqdm(df.groupby("ImageID"), desc=split, unit="img"):
        src_p = id2path.get(img_id)
        if not src_p:
            continue

        dst_img = img_dst_dir / src_p.name
        if not dst_img.exists():
            try:
                os.link(src_p, dst_img)
            except OSError:
                shutil.copyfile(src_p, dst_img)

        # Build YOLO label lines
        lines = []
        for _, row in group.iterrows():
            nm = mid2name.get(row.LabelName)
            if nm is None:
                continue
            cls_idx = name2idx.get(nm)
            if cls_idx is None:
                continue
            x0, x1 = row.XMin, row.XMax
            y0, y1 = row.YMin, row.YMax
            x_ctr  = (x0 + x1) / 2.0
            y_ctr  = (y0 + y1) / 2.0
            w      = x1 - x0
            h      = y1 - y0
            lines.append(f"{cls_idx} {x_ctr:.6f} {y_ctr:.6f} {w:.6f} {h:.6f}")

        # Write label file only if there are annotations
        if lines:
            (lbl_dst_dir / f"{img_id}.txt").write_text("\n".join(lines))

if __name__ == "__main__":
    for split in SPLITS:
        convert_split(split)
    print("\n✅ Conversion complete (fast)!")