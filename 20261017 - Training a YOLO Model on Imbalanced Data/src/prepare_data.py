"""Build the class-imbalance dataset variants for the GTSDB workshop.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import yaml


def write_yaml(path: Path, names: list[str], train: str, val: str) -> None:
    lines = [f"train: {train}", f"val: {val}", f"nc: {len(names)}", "names:"]
    for i, n in enumerate(names):
        lines.append(f"  {i}: {n}")
    path.write_text("\n".join(lines) + "\n")


def read_labels(labels_dir: Path, images_dir: Path | None = None) -> list[tuple[Path, Path, list[int]]]:
    rows = []
    for label_path in sorted(labels_dir.glob("*.txt")):
        instances = [int(line.split()[0]) for line in label_path.read_text().strip().splitlines() if line.strip()]
        if instances:
            img_path = images_dir / f"{label_path.stem}.jpg" if images_dir else label_path.with_suffix(".jpg")
            rows.append((img_path, label_path, instances))
    return rows


def count_classes(rows: list[tuple[Path, Path, list[int]]], nc: int) -> np.ndarray:
    counts = np.zeros(nc, dtype=int)
    for _, _, instances in rows:
        for c in instances:
            counts[c] += 1
    return counts


def build_oversample(data_root: Path, name: str, names: list[str],
                     suffix: str = "oversample") -> Path:
    n_classes = len(names)

    variant_yaml = data_root / f"{name}-{suffix}.yaml"
    if variant_yaml.exists():
        print(f"[ovs] already present at {variant_yaml} -> skip")
        return variant_yaml

    source_images = data_root / name / "images" / "train"
    source_labels = data_root / name / "labels" / "train"
    baseline_images = sorted(source_images.glob("*.jpg"))
    if not baseline_images:
        raise FileNotFoundError(
            f"no train images under {source_images}")

    variant_root = data_root / f"{name}-{suffix}"
    variant_images = variant_root / "images" / "train"
    variant_labels = variant_root / "labels" / "train"
    variant_images.mkdir(parents=True, exist_ok=True)
    variant_labels.mkdir(parents=True, exist_ok=True)

    # exact copy of the original training split
    for image_path in baseline_images:
        shutil.copyfile(image_path, variant_images / image_path.name)
        shutil.copyfile(source_labels / f"{image_path.stem}.txt",
                        variant_labels / f"{image_path.stem}.txt")

    # calculate how many more images we need to copy for each class
    image_records = read_labels(source_labels, source_images)
    instances_per_class = count_classes(image_records, n_classes)
    target_instances = int(instances_per_class.max())
    instances_missing = np.maximum(target_instances - instances_per_class, 0)
    instances_missing[instances_per_class == 0] = 0

    n_baseline_images = len(baseline_images)
    print(f"[ovs] base copied={n_baseline_images} target={target_instances}")

    # duplication
    images_containing_class = {c: [rec for rec in image_records if c in rec[2]]
                               for c in range(n_classes)}
    round_robin = {c: 0 for c in range(n_classes)}   # index into each class's pool
    source_images_used: set = set()
    n_duplicates = 0

    while instances_missing.max() > 0:
        class_id = int(instances_missing.argmax())      # furthest below target
        candidate_pool = images_containing_class[class_id]
        if not candidate_pool:
            break                                       # nothing left to copy for it
        source_image, source_label, classes_in_image = candidate_pool[
            round_robin[class_id] % len(candidate_pool)]
        round_robin[class_id] += 1

        source_images_used.add(source_image)
        duplicate_stem = f"ovs_{n_duplicates:06d}"
        shutil.copyfile(source_image, variant_images / f"{duplicate_stem}.jpg")
        shutil.copyfile(source_label, variant_labels / f"{duplicate_stem}.txt")
        n_duplicates += 1

        for other_class in set(classes_in_image):
            instances_missing[other_class] -= classes_in_image.count(other_class)

    print(f"[ovs] target={target_instances}: images {n_baseline_images} -> "
          f"{n_baseline_images + n_duplicates} "
          f"(+{n_duplicates} duplicates drawn from {len(source_images_used)} distinct source images)")

    write_yaml(variant_yaml, names, train=f"{name}-{suffix}/images/train",
               val=f"{name}/images/val")
    return variant_yaml


def build_rfs(data_root: Path, name: str, names: list[str], threshold: float = 0.015,
              target_rarest_repeat: int = 3) -> Path:
    """Repeat Factor Sampling as a train .txt path list.

    The same hypothesis as oversampling -- rare classes need more exposure -- but
    with a CEILING.

    For image i the repeat factor is r_i = max(1, sqrt(t / f_c)) over classes c in
    image i, where f_c is the FRACTION OF TRAIN IMAGES containing class c, and t is
    the threshold. 
    """
    n_classes = len(names)

    # outputs and guards 
    variant_yaml = data_root / f"{name}-rfs.yaml"
    train_list = data_root / f"{name}-rfs-train.txt"
    if variant_yaml.exists() and train_list.exists():
        print(f"[rfs] already present at {variant_yaml} -> skip")
        return variant_yaml

    source_images = data_root / name / "images" / "train"
    source_labels = data_root / name / "labels" / "train"

    # which classes appear in which image 
    classes_per_image: dict[str, list[int]] = {}
    for image_path in sorted(source_images.glob("*.jpg")):
        label_path = source_labels / f"{image_path.stem}.txt"
        class_ids: list[int] = []
        if label_path.exists():
            for line in label_path.read_text().strip().splitlines():
                if line.strip():
                    class_ids.append(int(line.split()[0]))
        classes_per_image[image_path.stem] = class_ids

    n_train_images = len(classes_per_image)
    if n_train_images == 0:
        raise FileNotFoundError(f"no train images under {source_images}")
    n_background_images = sum(1 for ids in classes_per_image.values() if not ids)

    # image frequency per class (f_c in the formula above) 
    images_containing_class = np.zeros(n_classes, dtype=np.int64)
    for class_ids in classes_per_image.values():
        for class_id in set(class_ids):
            images_containing_class[class_id] += 1
    image_fraction = images_containing_class.astype(np.float64) / n_train_images

    # derive t so the RAREST class lands on target_rarest_repeat 
    present_fractions = image_fraction[image_fraction > 0]
    rarest_image_fraction = float(present_fractions.min()) if present_fractions.size else 1.0
    if threshold is None:
        threshold = (target_rarest_repeat ** 2) * rarest_image_fraction

    # one path-list entry per copy 
    path_list: list[str] = []
    repeat_histogram: dict[int, int] = {}
    for image_stem, class_ids in classes_per_image.items():
        repeat_factor = 1.0
        for class_id in set(class_ids):
            if image_fraction[class_id] > 0:
                repeat_factor = max(repeat_factor,
                                    np.sqrt(threshold / image_fraction[class_id]))
        n_copies = max(1, int(round(repeat_factor)))
        repeat_histogram[n_copies] = repeat_histogram.get(n_copies, 0) + 1
        path_list.extend([(source_images / f"{image_stem}.jpg").as_posix()] * n_copies)

    # write 
    rng = np.random.default_rng(0)
    rng.shuffle(path_list)
    train_list.write_text("\n".join(path_list) + "\n", encoding="utf-8")

    print(f"[rfs] train images={n_train_images} (background-only={n_background_images}) classes={n_classes}")
    print(f"[rfs] f_min (rarest image freq)={rarest_image_fraction:.6f} -> threshold t={threshold:.6f}")
    print(f"[rfs] repeat histogram {dict(sorted(repeat_histogram.items()))} (#images by repeat count)")
    print(f"[rfs] list entries={len(path_list)} (unique images={n_train_images}) -> {train_list}")

    write_yaml(variant_yaml, names, train=train_list.name, val=f"{name}/images/val")
    return variant_yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="datasets", type=Path)
    ap.add_argument("--data-yaml", default="datasets/gtsdb_yolo.yaml", type=Path,
                    help="base dataset yaml: the source of the class names")
    ap.add_argument("--name", default="gtsdb_yolo")
    ap.add_argument("--oversample-suffix", default="oversample")
    ap.add_argument("--rfs-threshold", default=None, type=float)
    ap.add_argument("--rfs-target-rarest-repeat", default=3, type=int)
    ap.add_argument("--step", default="all", choices=["all", "oversample", "rfs"])
    args = ap.parse_args()

    ds = yaml.safe_load(args.data_yaml.read_text(encoding="utf-8"))
    names = [v for _, v in sorted(ds["names"].items(), key=lambda kv: int(kv[0]))]

    root = args.data_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.step in ("all", "oversample"):
        print(f"[data] OVERSAMPLE yaml: "
              f"{build_oversample(root, args.name, names, args.oversample_suffix)}")
    if args.step in ("all", "rfs"):
        print(f"[data] RFS yaml: "
              f"{build_rfs(root, args.name, names, args.rfs_threshold, args.rfs_target_rarest_repeat)}")


if __name__ == "__main__":
    main()
