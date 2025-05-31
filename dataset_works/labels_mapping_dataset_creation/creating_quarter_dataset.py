#!/usr/bin/env python3
import random
from pathlib import Path
import shutil
from tqdm import tqdm

# ───  PATHS ───────────────────────────────────────────
# Original merged YOLO dataset
ORIG_ROOT    = Path("/media/sameerhashmi/ran_epav_disk/Sameer_dataset_from_smb/merged_dataset")
# Path to write the 25%‐sized subset
OUT_ROOT     = Path("/media/sameerhashmi/ran_epav_disk/Sameer_dataset_from_smb/merged_dataset_quarter")
# My unified class list
MASTER_NAMES = Path("/media/sameerhashmi/ran_epav_disk/Sameer_dataset_from_smb/script/master.names")
# ────────────────────────────────────────────────────────────────

SPLITS       = ["train", "val"]
SAMPLE_RATIO = 0.25  # keep 25% of images, plus extras to cover all classes

# Load class names
names = [l.strip() for l in MASTER_NAMES.read_text().splitlines() if l.strip()]
nc = len(names)
print(f"Found {nc} classes in {MASTER_NAMES}\n")

for split in SPLITS:
    print(f"→ Processing split: {split}")
    img_src = ORIG_ROOT / "images" / split
    lbl_src = ORIG_ROOT / "labels" / split
    img_dst = OUT_ROOT / "images" / split
    lbl_dst = OUT_ROOT / "labels" / split
    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)

    # 1) Gather all images
    all_imgs = [p for p in img_src.iterdir() if p.is_file()]
    N = len(all_imgs)
    k = int(N * SAMPLE_RATIO)
    print(f"  Total images: {N}, initial sample size: {k}")

    # 2) Randomly pick 25%
    sampled = set(random.sample(all_imgs, k))

    # 3) Map each image → its set of class‐indices
    img2cls = {}
    for img in tqdm(all_imgs, desc="  Scanning labels", unit="img"):
        cls_set = set()
        lbl = lbl_src / f"{img.stem}.txt"
        if lbl.exists():
            for line in lbl.read_text().splitlines():
                if not line: continue
                cls_set.add(int(line.split()[0]))
        img2cls[img] = cls_set

    # 4) Compute which classes are present in the sample
    present = set().union(*(img2cls[i] for i in sampled))
    missing = set(range(nc)) - present
    print(f"  Classes covered: {len(present)} / {nc}, missing {len(missing)}")

    # 5) For each missing class, add one image that contains it
    for cls in missing:
        for img, cls_set in img2cls.items():
            if cls in cls_set and img not in sampled:
                sampled.add(img)
                break

    # 6) Final coverage
    present_final = set().union(*(img2cls[i] for i in sampled))
    print(f"  Final coverage: {len(present_final)} / {nc} classes")
    print(f"  Final sample size: {len(sampled)} images")

    # 7) Copy files
    copied = 0
    for img in tqdm(sampled, desc="  Copying files", unit="img"):
        shutil.copy2(img, img_dst / img.name)
        lbl = lbl_src / f"{img.stem}.txt"
        if lbl.exists():
            shutil.copy2(lbl, lbl_dst / lbl.name)
        copied += 1
    print(f"  Split '{split}': copied {copied} images + labels\n")

print("✅ Subsampling complete!")


# Output:

#   Scanning labels: 100%|████████████████████████████████████| 2276114/2276114 [13:34<00:00, 2795.83img/s]
#   Classes covered: 722 / 801, missing 79
#   Final coverage: 757 / 801 classes
#   Final sample size: 569063 images
#   Copying files: 100%|█████████████████████████████████████████| 569063/569063 [23:32<00:00, 402.93img/s]
#   Split 'train': copied 569063 images + labels

# → Processing split: val
#   Total images: 106309, initial sample size: 26577
#   Scanning labels: 100%|██████████████████████████████████████| 106309/106309 [00:49<00:00, 2135.17img/s]
#   Classes covered: 589 / 801, missing 212
#   Final coverage: 664 / 801 classes
#   Final sample size: 26646 images
#   Copying files: 100%|███████████████████████████████████████████| 26646/26646 [01:14<00:00, 355.58img/s]
#   Split 'val': copied 26646 images + labels

# ✅ Subsampling complete!