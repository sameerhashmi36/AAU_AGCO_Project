import json
import shutil
from pathlib import Path
from tqdm import tqdm

# === CONFIGURATION ===
# Path to the CrowdHuman root (contains annotation_train.odgt, annotation_val.odgt, and images/)
DATA_ROOT        = Path("/media/sameerhashmi/ran_epav_disk/Sameer_dataset_from_smb/data_sameer/crowdHuman")
# path to write the converted YOLOv11 dataset
OUTPUT_ROOT      = Path("/media/sameerhashmi/ran_epav_disk/Sameer_dataset_from_smb/converted_datasets/crowdHuman")
# Path to the master.names file
MASTER_NAMES_PTH = Path("./master.names")

# Splits mapping: split name -> odgt filename
SPLITS = {
    "train": "annotation_train.odgt",
    "val":   "annotation_val.odgt",
}

def normalize(name: str) -> str:
    return name.lower().strip().replace(" ", "_").replace("/", "_")

# Load master.names into a name->index map
master_names = [l.strip() for l in MASTER_NAMES_PTH.read_text().splitlines() if l.strip()]
name2idx = {normalize(n): i for i, n in enumerate(master_names)}

# Create output directories
for split in SPLITS:
    (OUTPUT_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)

total_images = 0
total_labels = 0

# Process each split
for split, odgt_fname in SPLITS.items():
    odgt_path = DATA_ROOT / odgt_fname
    img_src_dir = DATA_ROOT / "images"
    img_dst_dir = OUTPUT_ROOT / "images" / split
    lbl_dst_dir = OUTPUT_ROOT / "labels" / split

    split_images = 0
    split_labels = 0

    # Read and convert each line in the .odgt file
    with odgt_path.open('r') as f:
        for line in tqdm(f, desc=f"Converting {split}", unit="img"):
            data = json.loads(line)
            img_id = data["ID"]

            # Determine image file path
            src_img = img_src_dir / f"{img_id}.jpg"
            if not src_img.exists():
                src_img = img_src_dir / f"{img_id}.png"
                if not src_img.exists():
                    continue  # image file not found

            # Get image dimensions from annotation, fallback to PIL if needed
            w = data.get("img_w") or data.get("width")
            h = data.get("img_h") or data.get("height")
            if w is None or h is None:
                from PIL import Image
                with Image.open(src_img) as im:
                    w, h = im.size

            # Build YOLO label lines
            lines = []
            for box in data.get("gtboxes", []):
                if box.get("tag") != "person":
                    continue  # skip masks or other tags
                # Choose full-body box if available, else visible-region, else head
                if "fbox" in box:
                    x, y, bw, bh = box["fbox"]
                elif "vbox" in box:
                    x, y, bw, bh = box["vbox"]
                elif "hbox" in box:
                    x, y, bw, bh = box["hbox"]
                else:
                    continue

                # Normalize coordinates for YOLO
                x_center = (x + bw/2) / w
                y_center = (y + bh/2) / h
                w_norm = bw / w
                h_norm = bh / h

                cls_idx = name2idx.get("person")
                if cls_idx is None:
                    continue
                lines.append(f"{cls_idx} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

            if not lines:
                continue  # skip images without person boxes

            # Copy image to output
            dst_img = img_dst_dir / src_img.name
            shutil.copy2(src_img, dst_img)
            split_images += 1

            # Write label file
            (lbl_dst_dir / f"{img_id}.txt").write_text("\n".join(lines))
            split_labels += 1

    print(f"{split}: images written: {split_images}, labels written: {split_labels}")

    total_images += split_images
    total_labels += split_labels

# Overall summary
print(f"\nOverall: {total_images} images, {total_labels} labels written for CrowdHuman")
