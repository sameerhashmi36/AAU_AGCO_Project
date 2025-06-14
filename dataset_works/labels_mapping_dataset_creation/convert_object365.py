#!/usr/bin/env python3
import os
import shutil
from pathlib import Path

from tqdm import tqdm

# === CONFIGURATION ===
# Root of your Objects365 dataset
INPUT_ROOT   = Path("/media/sameerhashmi/ran_epav_disk/Sameer_dataset_from_smb/data_sameer/Objects365/")
# Where to write the converted YOLOv11 dataset
OUTPUT_ROOT  = Path("/media/sameerhashmi/ran_epav_disk/Sameer_dataset_from_smb/converted_datasets/")
# Path to your master.names file
MASTER_NAMES_PTH = Path("./master.names")

OBJECT_NAME = "Objects365"

# Load master.names → name2idx
master_names = [ln.strip() for ln in MASTER_NAMES_PTH.read_text().splitlines() if ln.strip()]
name2idx     = {name: idx for idx, name in enumerate(master_names)}

# Raw Objects365 class list (original order)
objects365_raw = [
    "Person","Sneakers","Chair","Other Shoes","Hat","Car","Lamp","Glasses","Bottle","Desk",
    "Cup","Street Lights","Cabinet/shelf","Handbag/Satchel","Bracelet","Plate","Picture/Frame",
    "Helmet","Book","Gloves","Storage box","Boat","Leather Shoes","Flower","Bench","Potted Plant",
    "Bowl/Basin","Flag","Pillow","Boots","Vase","Microphone","Necklace","Ring","SUV","Wine Glass",
    "Belt","Monitor/TV","Backpack","Umbrella","Traffic Light","Speaker","Watch","Tie",
    "Trash bin Can","Slippers","Bicycle","Stool","Barrel/bucket","Van","Couch","Sandals",
    "Basket","Drum","Pen/Pencil","Bus","Wild Bird","High Heels","Motorcycle","Guitar","Carpet",
    "Cell Phone","Bread","Camera","Canned","Truck","Traffic cone","Cymbal","Lifesaver","Towel",
    "Stuffed Toy","Candle","Sailboat","Laptop","Awning","Bed","Faucet","Tent","Horse","Mirror",
    "Power outlet","Sink","Apple","Air Conditioner","Knife","Hockey Stick","Paddle",
    "Pickup Truck","Fork","Traffic Sign","Balloon","Tripod","Dog","Spoon","Clock","Pot","Cow",
    "Cake","Dinning Table","Sheep","Hanger","Blackboard/Whiteboard","Napkin","Other Fish",
    "Orange/Tangerine","Toiletry","Keyboard","Tomato","Lantern","Machinery Vehicle","Fan",
    "Green Vegetables","Banana","Baseball Glove","Airplane","Mouse","Train","Pumpkin","Soccer",
    "Skiboard","Luggage","Nightstand","Tea pot","Telephone","Trolley","Head Phone","Sports Car",
    "Stop Sign","Dessert","Scooter","Stroller","Crane","Remote","Refrigerator","Oven","Lemon",
    "Duck","Baseball Bat","Surveillance Camera","Cat","Jug","Broccoli","Piano","Pizza","Elephant",
    "Skateboard","Surfboard","Gun","Skating and Skiing shoes","Gas stove","Donut","Bow Tie",
    "Carrot","Toilet","Kite","Strawberry","Other Balls","Shovel","Pepper","Computer Box",
    "Toilet Paper","Cleaning Products","Chopsticks","Microwave","Pigeon","Baseball",
    "Cutting/chopping Board","Coffee Table","Side Table","Scissors","Marker","Pie","Ladder",
    "Snowboard","Cookies","Radiator","Fire Hydrant","Basketball","Zebra","Grape","Giraffe",
    "Potato","Sausage","Tricycle","Violin","Egg","Fire Extinguisher","Candy","Fire Truck",
    "Billiards","Converter","Bathtub","Wheelchair","Golf Club","Briefcase","Cucumber",
    "Cigar/Cigarette","Paint Brush","Pear","Heavy Truck","Hamburger","Extractor","Extension Cord",
    "Tong","Tennis Racket","Folder","American Football","Earphone","Mask","Kettle","Tennis",
    "Ship","Swing","Coffee Machine","Slide","Carriage","Onion","Green beans","Projector",
    "Frisbee","Washing Machine/Drying Machine","Chicken","Printer","Watermelon","Saxophone",
    "Tissue","Toothbrush","Ice cream","Hot-air balloon","Cello","French Fries","Scale","Trophy",
    "Cabbage","Hot dog","Blender","Peach","Rice","Wallet/Purse","Volleyball","Deer","Goose",
    "Tape","Tablet","Cosmetics","Trumpet","Pineapple","Golf Ball","Ambulance","Parking meter",
    "Mango","Key","Hurdle","Fishing Rod","Medal","Flute","Brush","Penguin","Megaphone","Corn",
    "Lettuce","Garlic","Swan","Helicopter","Green Onion","Sandwich","Nuts","Speed Limit Sign",
    "Induction Cooker","Broom","Trombone","Plum","Rickshaw","Goldfish","Kiwi fruit",
    "Router/modem","Poker Card","Toaster","Shrimp","Sushi","Cheese","Notepaper","Cherry","Pliers",
    "CD","Pasta","Hammer","Cue","Avocado","Hamimelon","Flask","Mushroom","Screwdriver","Soap",
    "Recorder","Bear","Eggplant","Board Eraser","Coconut","Tape Measure/Ruler","Pig","Showerhead",
    "Globe","Chips","Steak","Crosswalk Sign","Stapler","Camel","Formula 1","Pomegranate",
    "Dishwasher","Crab","Hoverboard","Meat ball","Rice Cooker","Tuba","Calculator","Papaya",
    "Antelope","Parrot","Seal","Butterfly","Dumbbell","Donkey","Lion","Urinal","Dolphin",
    "Electric Drill","Hair Dryer","Egg tart","Jellyfish","Treadmill","Lighter","Grapefruit",
    "Game board","Mop","Radish","Baozi","Target","French","Spring Rolls","Monkey","Rabbit",
    "Pencil Case","Yak","Red Cabbage","Binoculars","Asparagus","Barbell","Scallop","Noddles",
    "Comb","Dumpling","Oyster","Table Tennis paddle","Cosmetics Brush/Eyeliner Pencil",
    "Chainsaw","Eraser","Lobster","Durian","Okra","Lipstick","Cosmetics Mirror","Curling",
    "Table Tennis"
]


def normalize(name: str) -> str:
    return name.lower().strip().replace(" ", "_").replace("/", "_")

objects365_names = [normalize(n) for n in objects365_raw]

# Ensure output dirs exist
for split in ("train", "val"):
    (OUTPUT_ROOT / OBJECT_NAME / "images" / split).mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / OBJECT_NAME / "labels" / split).mkdir(parents=True, exist_ok=True)

# Process each split
for split in ("train", "val"):
    img_src = INPUT_ROOT / "images" / split
    lbl_src = INPUT_ROOT / "labels" / split
    img_dst = OUTPUT_ROOT / OBJECT_NAME / "images" / split
    lbl_dst = OUTPUT_ROOT / OBJECT_NAME / "labels" / split

    # Copy images (skip hidden files)
    for img_path in tqdm(img_src.iterdir(), desc=f"{split} images", unit="img"):
        if not img_path.is_file() or img_path.name.startswith('.'):
            continue
        dst = img_dst / img_path.name
        try:
            os.link(img_path, dst)
        except OSError:
            shutil.copy2(img_path, dst)

    # Remap labels (skip hidden .txt files)
    for lbl_path in tqdm(lbl_src.iterdir(), desc=f"{split} labels", unit="file"):
        if not lbl_path.is_file() or lbl_path.name.startswith('.'):
            continue
        # ensure it's a .txt
        if lbl_path.suffix.lower() != '.txt':
            continue

        lines = []
        for row in lbl_path.read_text().splitlines():
            parts = row.split()
            orig_cls = int(parts[0])
            if 0 <= orig_cls < len(objects365_names):
                nm = objects365_names[orig_cls]
                new_idx = name2idx.get(nm)
                if new_idx is not None:
                    lines.append(" ".join([str(new_idx)] + parts[1:]))
        if lines:
            (lbl_dst / lbl_path.name).write_text("\n".join(lines))

print("✅ Objects365 conversion complete—hidden files skipped!")