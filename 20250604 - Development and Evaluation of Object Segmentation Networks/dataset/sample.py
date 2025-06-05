import os
import random
import shutil
import json
from pycocotools.coco import COCO

def sample_images_split_3way(
    source_dir,
    dest_base_dir,
    annotation_file,
    train_size,
    val_size,
    test_size,
    target_classes,
    extensions=None
):
    if extensions is None:
        extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')

    # Output dirs
    split_dirs = {split: os.path.join(dest_base_dir, split) for split in ["train", "val", "test"]}
    for path in split_dirs.values():
        os.makedirs(path, exist_ok=True)

    # Load full COCO annotations
    with open(annotation_file, 'r') as f:
        coco_data = json.load(f)

    coco = COCO(annotation_file)
    cat_ids = coco.getCatIds(catNms=target_classes)
    anns = coco.loadAnns(coco.getAnnIds(catIds=cat_ids))
    img_ids = list({ann['image_id'] for ann in anns})
    imgs_info = coco.loadImgs(img_ids)

    valid_imgs = [
        img for img in imgs_info
        if os.path.exists(os.path.join(source_dir, img['file_name']))
        and img['file_name'].lower().endswith(extensions)
    ]

    total_required = train_size + val_size + test_size
    if total_required > len(valid_imgs):
        raise ValueError(f"Requested {total_required} images, but only {len(valid_imgs)} found.")

    # Shuffle and split
    random.shuffle(valid_imgs)
    split_data = {
        "train": valid_imgs[:train_size],
        "val": valid_imgs[train_size:train_size + val_size],
        "test": valid_imgs[train_size + val_size:train_size + val_size + test_size]
    }

    # Helper to build full annotation structure like original
    def write_coco_subset(images, split_name):
        image_ids = {img["id"] for img in images}
        annotations = [ann for ann in coco_data["annotations"] if ann["image_id"] in image_ids]

        subset = {
            "info": coco_data.get("info", {}),
            "licenses": coco_data.get("licenses", []),
            "images": images,
            "annotations": annotations,
            "categories": [cat for cat in coco_data["categories"] if cat["id"] in cat_ids]
        }

        with open(os.path.join(dest_base_dir, f"{split_name}.json"), "w") as f:
            json.dump(subset, f, indent=4)

    # Copy images and write annotation files
    for split, imgs in split_data.items():
        for img in imgs:
            shutil.copy2(
                os.path.join(source_dir, img['file_name']),
                os.path.join(split_dirs[split], img['file_name'])
            )
        write_coco_subset(imgs, split)

    print(f"Copied and annotated: {train_size} train, {val_size} val, {test_size} test images.")

# Example usage
if __name__ == "__main__":
    source_directory = "./source"
    destination_base = "./coco_sample"
    annotation_path = "./annotations/instances_val2017.json"
    train_sample_size = 275
    val_sample_size = 28
    test_sample_size = 45
    desired_classes = ["dog", "cat", "person", "chair", "mouse", "remote", "keyboard", "cell phone", "cup", "fork", "knife", "spoon"]

    sample_images_split_3way(
        source_directory,
        destination_base,
        annotation_path,
        train_sample_size,
        val_sample_size,
        test_sample_size,
        desired_classes
    )
