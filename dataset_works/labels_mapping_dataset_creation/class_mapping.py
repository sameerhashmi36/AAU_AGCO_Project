# Master class mapping for COCO, Objects365, OpenImagesV6, VOC2012
import json
import pandas as pd
from pathlib import Path

def normalize(name: str) -> str:
    return name.lower().strip().replace(' ', '_').replace('/', '_')

# 1) Hardcode COCO class names (80)
coco_names = [
    'person','bicycle','car','motorcycle','airplane','bus','train','truck',
    'boat','traffic_light','fire_hydrant','stop_sign','parking_meter','bench',
    'bird','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe',
    'backpack','umbrella','handbag','tie','suitcase','frisbee','skis','snowboard',
    'sports_ball','kite','baseball_bat','baseball_glove','skateboard','surfboard',
    'tennis_racket','bottle','wine_glass','cup','fork','knife','spoon','bowl',
    'banana','apple','sandwich','orange','broccoli','carrot','hot_dog','pizza',
    'donut','cake','chair','couch','potted_plant','bed','dining_table','toilet',
    'tv','laptop','mouse','remote','keyboard','cell_phone','microwave','oven',
    'toaster','sink','refrigerator','book','clock','vase','scissors',
    'teddy_bear','hair_drier','toothbrush'
]

# 2) Hardcode VOC2012 class names (20)
voc_names = [
    'aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow',
    'diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa',
    'train','tvmonitor'
]

# 3) Hardcode Objects365 class names (365)
objects365_names = [
    "person","sneakers","chair","other_shoes","hat","car","lamp","glasses","bottle",
    "desk","cup","street_lights","cabinet_shelf","handbag_satchel","bracelet","plate",
    "picture_frame","helmet","book","gloves","storage_box","boat","leather_shoes",
    "flower","bench","potted_plant","bowl_basin","flag","pillow","boots","vase",
    "microphone","necklace","ring","suv","wine_glass","belt","monitor_tv","backpack",
    "umbrella","traffic_light","speaker","watch","tie","trash_bin_can","slippers",
    "bicycle","stool","barrel_bucket","van","couch","sandals","basket","drum",
    "pen_pencil","bus","wild_bird","high_heels","motorcycle","guitar","carpet",
    "cell_phone","bread","camera","canned","truck","traffic_cone","cymbal","lifesaver",
    "towel","stuffed_toy","candle","sailboat","laptop","awning","bed","faucet","tent",
    "horse","mirror","power_outlet","sink","apple","air_conditioner","knife",
    "hockey_stick","paddle","pickup_truck","fork","traffic_sign","balloon","tripod",
    "dog","spoon","clock","pot","cow","cake","dinning_table","sheep","hanger",
    "blackboard_whiteboard","napkin","other_fish","orange_tangerine","toiletry",
    "keyboard","tomato","lantern","machinery_vehicle","fan","green_vegetables",
    "banana","baseball_glove","airplane","mouse","train","pumpkin","soccer",
    "skiboard","luggage","nightstand","tea_pot","telephone","trolley","head_phone",
    "sports_car","stop_sign","dessert","scooter","stroller","crane","remote",
    "refrigerator","oven","lemon","duck","baseball_bat","surveillance_camera","cat",
    "jug","broccoli","piano","pizza","elephant","skateboard","surfboard","gun",
    "skating_and_skiing_shoes","gas_stove","donut","bow_tie","carrot","toilet","kite",
    "strawberry","other_balls","shovel","pepper","computer_box","toilet_paper",
    "cleaning_products","chopsticks","microwave","pigeon","baseball","cutting_chopping_board",
    "coffee_table","side_table","scissors","marker","pie","ladder","snowboard","cookies",
    "radiator","fire_hydrant","basketball","zebra","grape","giraffe","potato","sausage",
    "tricycle","violin","egg","fire_extinguisher","candy","fire_truck","billiards",
    "converter","bathtub","wheelchair","golf_club","briefcase","cucumber","cigar_cigarette",
    "paint_brush","pear","heavy_truck","hamburger","extractor","extension_cord","tong",
    "tennis_racket","folder","american_football","earphone","mask","kettle","tennis",
    "ship","swing","coffee_machine","slide","carriage","onion","green_beans",
    "projector","frisbee","washing_machine_drying_machine","chicken","printer",
    "watermelon","saxophone","tissue","toothbrush","ice_cream","hot_air_balloon",
    "cello","french_fries","scale","trophy","cabbage","hot_dog","blender","peach","rice",
    "wallet_purse","volleyball","deer","goose","tape","tablet","cosmetics","trumpet",
    "pineapple","golf_ball","ambulance","parking_meter","mango","key","hurdle","fishing_rod",
    "medal","flute","brush","penguin","megaphone","corn","lettuce","garlic","swan","helicopter",
    "green_onion","sandwich","nuts","speed_limit_sign","induction_cooker","broom","trombone",
    "plum","rickshaw","goldfish","kiwi_fruit","router_modem","poker_card","toaster","shrimp","sushi",
    "cheese","notepaper","cherry","pliers","cd","pasta","hammer","cue","avocado","hamimelon","flask",
    "mushroom","screwdriver","soap","recorder","bear","eggplant","board_eraser","coconut",
    "tape_measure_ruler","pig","showerhead","globe","chips","steak","crosswalk_sign","stapler",
    "camel","formula_1","pomegranate","dishwasher","crab","hoverboard","meat_ball","rice_cooker",
    "tuba","calculator","papaya","antelope","parrot","seal","butterfly","dumbbell","donkey","lion",
    "urinal","dolphin","electric_drill","hair_dryer","egg_tart","jellyfish","treadmill","lighter",
    "grapefruit","game_board","mop","radish","baozi","target","french","spring_rolls","monkey",
    "rabbit","pencil_case","yak","red_cabbage","binoculars","asparagus","barbell","scallop","noddles",
    "comb","dumpling","oyster","table_tennis_paddle","cosmetics_brush_eyeliner_pencil","chainsaw",
    "eraser","lobster","durian","okra","lipstick","cosmetics_mirror","curling","table_tennis"  # last object
]

# 4) Load Open Images V6 classes from CSV (no header)
oi_csv = Path("/media/sameerhashmi/ran_epav_disk/Sameer_dataset_from_smb/data_sameer/open-images-v6/train/metadata/classes.csv")
oi_df = pd.read_csv(oi_csv, header=None, names=["LabelName","DisplayName"])
oi_names = [normalize(n) for n in oi_df["DisplayName"].tolist()]

# 5) Build master list in order: COCO → Objects365 → OpenImages → VOC → (CrowdHuman if needed)
master = []
for names in (coco_names, objects365_names, oi_names, voc_names):
    for n in names:
        if n not in master:
            master.append(n)

# Print summary
print(f"Master label space size: {len(master)} classes")
# Optionally save
Path("./master.names").write_text("\n".join(master))
