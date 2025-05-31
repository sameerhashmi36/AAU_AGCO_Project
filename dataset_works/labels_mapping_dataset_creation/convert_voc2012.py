#!/usr/bin/env python3
import random
import shutil
from pathlib import Path
import xml.etree.ElementTree as ET

# === PATHs ===
random.seed(42)  # for reproducibility
VOC_ROOT     = Path("/media/sameerhashmi/ran_epav_disk/Sameer_dataset_from_smb/data_sameer/voc2012")
OUTPUT_ROOT  = Path("/media/sameerhashmi/ran_epav_disk/Sameer_dataset_from_smb/converted_datasets/voc2012_split")
MASTER_NAMES = Path("./master.names")  # adjust path as needed

SPLIT_RATIO = 0.8  # 80% train, 20% val

def normalize(name: str) -> str:
    return name.lower().strip().replace(" ", "_")

def load_master(pth: Path):
    return {
        normalize(line): idx
        for idx, line in enumerate(pth.read_text().splitlines())
        if line.strip()
    }

# load mapping
name2idx = load_master(MASTER_NAMES)

# prepare output directories
for split in ("train", "val"):
    (OUTPUT_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)

# collect all valid image-label pairs
samples = []
ann_dir = VOC_ROOT / "Annotations"
img_dir = VOC_ROOT / "images"

for xml_file in ann_dir.glob("*.xml"):
    stem = xml_file.stem
    # find corresponding image
    img_file = None
    for ext in (".jpg", ".jpeg", ".png"):
        cand = img_dir / f"{stem}{ext}"
        if cand.exists():
            img_file = cand
            break
    if img_file is None:
        continue

    # parse xml
    tree = ET.parse(xml_file)
    root = tree.getroot()
    size = root.find("size")
    w = float(size.find("width").text)
    h = float(size.find("height").text)

    yolo_lines = []
    for obj in root.findall("object"):
        cls = normalize(obj.find("name").text)
        if cls not in name2idx:
            continue
        idx = name2idx[cls]
        b = obj.find("bndbox")
        xmin = float(b.find("xmin").text)
        ymin = float(b.find("ymin").text)
        xmax = float(b.find("xmax").text)
        ymax = float(b.find("ymax").text)
        x_center = ((xmin + xmax) / 2) / w
        y_center = ((ymin + ymax) / 2) / h
        bw = (xmax - xmin) / w
        bh = (ymax - ymin) / h
        yolo_lines.append(f"{idx} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}")

    if not yolo_lines:
        continue

    samples.append((stem, img_file, yolo_lines))

# shuffle and split
random.shuffle(samples)
n_train = int(len(samples) * SPLIT_RATIO)
train_samples = samples[:n_train]
val_samples   = samples[n_train:]

# helper to write a batch
def write_samples(batch, split):
    img_out = OUTPUT_ROOT / "images" / split
    lbl_out = OUTPUT_ROOT / "labels" / split
    for stem, img_file, yolo_lines in batch:
        # copy image
        shutil.copy2(img_file, img_out / img_file.name)
        # write label
        (lbl_out / f"{stem}.txt").write_text("\n".join(yolo_lines))

# write train and val
write_samples(train_samples, "train")
write_samples(val_samples,   "val")

# prints
print(f"Total images: {len(samples)}")
print(f"Train: {len(train_samples)} images")
print(f"Val:   {len(val_samples)} images")
