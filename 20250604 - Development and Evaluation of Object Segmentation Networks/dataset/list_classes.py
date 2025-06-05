from pycocotools.coco import COCO

# Path to the annotation JSON (e.g., for val2017)
annotation_file = "./annotations/instances_val2017.json"

# Load COCO annotations
coco = COCO(annotation_file)

# Get all category names
categories = coco.loadCats(coco.getCatIds())
category_names = [cat['name'] for cat in categories]

print("Available COCO classes:")
for name in category_names:
    print("-", name)
