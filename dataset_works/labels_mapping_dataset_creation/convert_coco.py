import os
import json
import shutil
from tqdm import tqdm

# CONFIGURE THESE
COCO_ROOT = "/media/sameerhashmi/ran_epav_disk/Sameer_dataset_from_smb/data_sameer/coco"
YOLO_ROOT = "/media/sameerhashmi/ran_epav_disk/Sameer_dataset_from_smb/converted_datasets/coco"

def convert_split(split):
    # Paths
    ann_path    = os.path.join(COCO_ROOT, "annotations", f"instances_{split}.json")
    img_src_dir = os.path.join(COCO_ROOT, "images", split)
    img_dst_dir = os.path.join(YOLO_ROOT, "images", split)
    lbl_dst_dir = os.path.join(YOLO_ROOT, "labels", split)

    os.makedirs(img_dst_dir, exist_ok=True)
    os.makedirs(lbl_dst_dir, exist_ok=True)

    # Load COCO file
    with open(ann_path, 'r') as f:
        coco = json.load(f)

    # Build image‐info and category‐mapping tables
    images  = {img["id"]:(img["file_name"], img["width"], img["height"])
               for img in coco["images"]}
    cats    = sorted(coco["categories"], key=lambda c: c["id"])
    cat2idx = {c["id"]:i for i,c in enumerate(cats)}

    # Group annotations by image_id
    ann_by_img = {}
    for ann in coco["annotations"]:
        ann_by_img.setdefault(ann["image_id"], []).append(ann)

    # Process each image with a progress bar
    for img_id, (fn, w, h) in tqdm(images.items(),
                                   desc=f"Converting {split}",
                                   unit="img"):
        # 1) copy image
        src = os.path.join(img_src_dir, fn)
        dst = os.path.join(img_dst_dir, fn)
        shutil.copy2(src, dst)

        # 2) write label file
        lines = []
        for ann in ann_by_img.get(img_id, []):
            x_min, y_min, bw, bh = ann["bbox"]
            x_ctr = (x_min + bw/2) / w
            y_ctr = (y_min + bh/2) / h
            w_norm = bw / w
            h_norm = bh / h
            cls_idx = cat2idx[ann["category_id"]]
            lines.append(f"{cls_idx} {x_ctr:.6f} {y_ctr:.6f} {w_norm:.6f} {h_norm:.6f}")

        label_path = os.path.splitext(fn)[0] + ".txt"
        with open(os.path.join(lbl_dst_dir, label_path), "w") as f:
            f.write("\n".join(lines))

if __name__ == "__main__":
    for split in ["train2017", "val2017"]:
        convert_split(split)
    print("✅ Conversion complete!")