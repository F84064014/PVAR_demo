from pathlib import Path

ATTRIBUTES = [
    "Age-Young", "Age-Adult", "Age-Old",
    "Gender-Male",
    "Gender-Female",
    "UpperBody-Color-Black", "UpperBody-Color-Blue",   "UpperBody-Color-Brown",
    "UpperBody-Color-Green", "UpperBody-Color-Grey",   "UpperBody-Color-Orange",
    "UpperBody-Color-Pink",  "UpperBody-Color-Purple", "UpperBody-Color-Red",
    "UpperBody-Color-White", "UpperBody-Color-Yellow",
    "LowerBody-Color-Black", "LowerBody-Color-Blue", "LowerBody-Color-Brown",
    "LowerBody-Color-Green", "LowerBody-Color-Grey", "LowerBody-Color-Orange",
    "LowerBody-Color-Pink", "LowerBody-Color-Purple", "LowerBody-Color-Red",
    "LowerBody-Color-White", "LowerBody-Color-Yellow",
    "Accessory-Backpack", "Accessory-Bag", "Accessory-Hat"
]

def getGroup():
    getGroupName = lambda s: '-'.join(s.split('-')[:-1])
    groups = set([getGroupName(attr) for attr in ATTRIBUTES])
    groups = {
        group: [attr for attr in ATTRIBUTES if group in attr] for group in groups
    }
    group_order = ["Age", "Gender", "UpperBody-Color", "LowerBody-Color", "Accessory"]
    groups = {group: groups[group] for group in group_order}
    return groups
GRUOPS = getGroup()

DEMO_IMGS = [str(p) for p in Path("samples/par").glob('*')]

DEMO_MODELS = list(map(str, (Path(__file__).parent.parent.parent / 'models' / 'par').iterdir()))
DEMO_MODEL = DEMO_MODELS[0]

DRESS_COLOR = {
    0:  (  0,   0,   0),    # Black
    1:  (255,   0,   0),    # Blue
    2:  ( 42,  42, 165),    # Brown
    3:  (  0, 128,   0),    # Green
    4:  (128, 128, 128),    # Grey
    5:  (  0, 165, 255),    # Orange
    6:  (203, 192, 255),    # Pink
    7:  (128,   0, 128),    # Purple
    8:  (  0,   0, 255),    # Red
    9:  (255, 255, 255),    # White
    10: (  0, 255, 255),    # Yellow
}

AGE_COLOR = {
    0: (0, 255, 0),
    1: (255, 0, 0),
    2: (0, 0, 255)
}