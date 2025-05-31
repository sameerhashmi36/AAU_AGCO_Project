from pathlib import Path
from tqdm import tqdm

# Base path of your converted Objects365 dataset
BASE = Path("/media/sameerhashmi/ran_epav_disk/Sameer_dataset_from_smb/converted_datasets/Objects365")

splits = ["train", "val"]

total_images_kept = 0
total_labels_kept = 0

for split in splits:
    img_dir = BASE / "images" / split
    lbl_dir = BASE / "labels" / split

    # Gather non-hidden stems
    image_stems = {p.stem for p in img_dir.iterdir() if p.is_file() and not p.name.startswith('.')}
    label_stems = {p.stem for p in lbl_dir.iterdir() if p.is_file() and p.suffix == ".txt" and not p.name.startswith('.')}

    # Find paired stems
    paired = image_stems & label_stems

    # Remove unpaired images
    for stem in tqdm(image_stems - paired, desc=f"{split} remove images", unit="img"):
        for file in img_dir.glob(f"{stem}.*"):
            file.unlink(missing_ok=True)

    # Remove unpaired labels
    for stem in tqdm(label_stems - paired, desc=f"{split} remove labels", unit="file"):
        (lbl_dir / f"{stem}.txt").unlink(missing_ok=True)

    # Counts after pruning
    images_count = len(paired)
    labels_count = len(paired)  # same as images_count

    print(f"{split}: {images_count} images, {labels_count} labels kept")

    total_images_kept += images_count
    total_labels_kept += labels_count

# Overall summary
print(f"Overall: {total_images_kept} images, {total_labels_kept} labels kept")