import shutil
from pathlib import Path
from tqdm import tqdm

# === PATHs ===
# Root of the individual converted YOLO datasets
INPUT_ROOT  = Path("/media/sameerhashmi/ran_epav_disk/Sameer_dataset_from_smb/converted_datasets")
# Destination for the merged dataset
MERGED_ROOT = Path("/media/sameerhashmi/ran_epav_disk/Sameer_dataset_from_smb/merged_dataset")
SPLITS      = ["train", "val"]

# Ensure merged directories exist
for split in SPLITS:
    (MERGED_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
    (MERGED_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)

# Track overall counts
overall_counts = {split: {"images": 0, "labels": 0} for split in SPLITS}

# Iterate each dataset folder under INPUT_ROOT
for dataset_dir in sorted(INPUT_ROOT.iterdir()):
    if not dataset_dir.is_dir():
        continue
    ds_name = dataset_dir.name
    print(f"\nDataset '{ds_name}':")
    for split in SPLITS:
        img_src = dataset_dir / "images" / split
        lbl_src = dataset_dir / "labels" / split
        img_dst = MERGED_ROOT / "images" / split
        lbl_dst = MERGED_ROOT / "labels" / split

        img_count = 0
        lbl_count = 0

        if img_src.exists() and lbl_src.exists():
            # Copy images and their labels
            for img_path in tqdm(img_src.iterdir(), desc=f"  {split} images", unit="img"):
                if not img_path.is_file():
                    continue
                new_img_name = f"{ds_name}_{img_path.name}"
                shutil.copy2(img_path, img_dst / new_img_name)
                img_count += 1
                overall_counts[split]["images"] += 1

                lbl_path = lbl_src / f"{img_path.stem}.txt"
                if lbl_path.exists():
                    new_lbl_name = f"{ds_name}_{img_path.stem}.txt"
                    shutil.copy2(lbl_path, lbl_dst / new_lbl_name)
                    lbl_count += 1
                    overall_counts[split]["labels"] += 1

        print(f"  {split}: {img_count} images copied, {lbl_count} labels copied")

# Print overall merged counts
print("\nOverall merged counts:")
for split in SPLITS:
    imgs = overall_counts[split]["images"]
    lbls = overall_counts[split]["labels"]
    print(f"  {split}: {imgs} images, {lbls} labels")