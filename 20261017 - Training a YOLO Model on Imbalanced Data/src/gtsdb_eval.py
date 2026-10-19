
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import yaml

DATA_YAML = Path(__file__).resolve().parent.parent / "datasets" / "gtsdb_yolo.yaml"
if not DATA_YAML.exists():
    raise FileNotFoundError(f"dataset yaml not found: {DATA_YAML}")
_DATASET = yaml.safe_load(DATA_YAML.read_text(encoding="utf-8"))

NAMES = [v for _, v in sorted(_DATASET["names"].items(), key=lambda kv: int(kv[0]))]
NC = len(NAMES)

HEAD_MIN = 21          # >= this many train instances -> head group
TAIL_MAX = 20          # <= this many train instances -> tail group
DETECT_CONF = 0.001   


def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def load_train_counts(data_root: Path) -> np.ndarray:
    counts = np.zeros(NC, dtype=int)
    for lbl in (data_root / "gtsdb_yolo" / "labels" / "train").glob("*.txt"):
        for line in lbl.read_text().strip().splitlines():
            if line.strip():
                counts[int(line.split()[0])] += 1
    return counts


def group_masks(counts: np.ndarray, head_min: int = HEAD_MIN, tail_max: int = TAIL_MAX) -> dict:
    return {
        "head": counts >= head_min,
        "tail": counts <= tail_max,
    }


def aggregate_groups(ap_class_index, p, r, ap50, ap, counts, head_min=HEAD_MIN, tail_max=TAIL_MAX) -> dict:
    """Mean validator metrics per head/tail group."""
    idx = np.asarray(ap_class_index, dtype=int)
    p = np.asarray(p, dtype=float)
    r = np.asarray(r, dtype=float)
    ap50 = np.asarray(ap50, dtype=float)
    ap = np.asarray(ap, dtype=float)

    masks = group_masks(counts, head_min=head_min, tail_max=tail_max)
    selections = {"overall": np.ones(idx.shape, dtype=bool)}
    for name, mask in masks.items():
        selections[name] = mask[idx] if idx.size else np.zeros(0, dtype=bool)

    def _mean(values, sel):
        return float(values[sel].mean()) if sel.size else 0.0

    out = {}
    for name, sel in selections.items():
        out[name] = {
            "mAP50": _mean(ap50, sel),
            "mAP50-95": _mean(ap, sel),
            "Precision": _mean(p, sel),
            "Recall": _mean(r, sel),
            "classes": int(sel.sum()),
        }
    return out


def run_val(ckpt: Path, data_yaml: Path, imgsz: int, device: str, conf: float,
            plots_out: Path | None = None):
    from ultralytics import YOLO

    tmp = Path(tempfile.mkdtemp(prefix="gtsdb_val_"))
    try:
        model = YOLO(str(ckpt))
        metrics = model.val(
            data=str(data_yaml),
            imgsz=imgsz,
            device=device,
            conf=conf,
            plots=plots_out is not None,
            save_json=False,
            verbose=False,
            project=str(tmp),
            name="val",
            exist_ok=True,
        )
        if plots_out is not None:
            out = Path(plots_out)
            out.mkdir(parents=True, exist_ok=True)
            for pattern in ("*.png", "*.jpg"):
                for src in (tmp / "val").glob(pattern):
                    shutil.copy(src, out / src.name)
        return metrics
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def per_class_records(box, nt_per_class, counts, head_min=HEAD_MIN) -> dict:
    """Per-class AP50/AP/precision/recall keyed by class name, tagged head/tail."""
    idx = np.asarray(box.ap_class_index, dtype=int) if len(box.ap_class_index) else np.zeros(0, dtype=int)
    present = {}
    if idx.size:
        p = np.asarray(box.p, dtype=float)
        r = np.asarray(box.r, dtype=float)
        ap50 = np.asarray(box.ap50, dtype=float)
        ap = np.asarray(box.ap, dtype=float)
        for i, c in enumerate(idx):
            present[int(c)] = (float(ap50[i]), float(ap[i]), float(p[i]), float(r[i]))

    records = {}
    for c in range(NC):
        group = "head" if counts[c] >= head_min else "tail"
        if c in present:
            ap50, ap, p, r = present[c]
            records[NAMES[c]] = {
                "AP50": ap50,
                "AP": ap,
                "precision": p,
                "recall": r,
                "gt_val": int(nt_per_class[c]) if nt_per_class is not None else 0,
                "n_train": int(counts[c]),
                "group": group,
            }
        else:
            records[NAMES[c]] = {
                "AP50": None,
                "AP": None,
                "precision": None,
                "recall": None,
                "gt_val": int(nt_per_class[c]) if nt_per_class is not None else 0,
                "n_train": int(counts[c]),
                "group": group,
            }
    return records


def evaluate(*, data_root: Path = Path("datasets"), data: Path | None = None,
             runs_root: Path = Path("runs"), device: str = "0", imgsz: int = 1280,
             only: str | None = None, no_plots: bool = False) -> dict:
    data_root = data_root.expanduser().resolve()
    runs_root = runs_root.expanduser().resolve()
    data_yaml = (data.expanduser().resolve() if data
                 else data_root / "gtsdb_yolo.yaml")
    if not data_yaml.exists():
        raise FileNotFoundError(f"dataset yaml not found: {data_yaml} (run the notebook setup first)")

    counts = load_train_counts(data_root)
    masks = group_masks(counts, head_min=HEAD_MIN, tail_max=TAIL_MAX)

    arms = sorted(p for p in runs_root.iterdir() if p.is_dir() and (p / "best.pt").exists())
    arms = [p for p in arms if not p.name.startswith(".")]
    if only:
        keep = set(only.split(","))
        arms = [p for p in arms if p.name in keep]

    summary = {"names": NAMES, "head_min": HEAD_MIN, "tail_max": TAIL_MAX,
               "train_counts": counts.tolist(),
               "groups": {k: [int(i) for i in np.where(m)[0]] for k, m in masks.items()}}

    for arm in arms:
        ckpt = arm / "best.pt"
        provenance = {"checkpoint": ckpt.name, "sha256": file_sha256(ckpt), "imgsz": imgsz,
                      "validator": "ultralytics"}
        print(f"[eval] {arm.name}: validating best.pt sha256={provenance['sha256'][:12]} "
              f"at imgsz={imgsz} with the Ultralytics validator ...", flush=True)
        plots_out = None if no_plots else arm / "eval_plots"
        metrics = run_val(ckpt, data_yaml, imgsz, device, DETECT_CONF, plots_out=plots_out)
        box = metrics.box
        agg = aggregate_groups(box.ap_class_index, box.p, box.r, box.ap50, box.ap,
                                counts, head_min=HEAD_MIN, tail_max=TAIL_MAX)
        nt_per_class = getattr(metrics, "nt_per_class", None)
        per_class = per_class_records(box, nt_per_class, counts, head_min=HEAD_MIN)

        payload = {
            "exp": arm.name,
            "schema": 2,
            "checkpoint": provenance,
            "AP": agg["overall"]["mAP50-95"],
            "AP50": agg["overall"]["mAP50"],
            "precision": agg["overall"]["Precision"],
            "recall": agg["overall"]["Recall"],
            "head_AP": agg["head"]["mAP50-95"],
            "head_AP50": agg["head"]["mAP50"],
            "head_precision": agg["head"]["Precision"],
            "head_recall": agg["head"]["Recall"],
            "tail_AP": agg["tail"]["mAP50-95"],
            "tail_AP50": agg["tail"]["mAP50"],
            "tail_precision": agg["tail"]["Precision"],
            "tail_recall": agg["tail"]["Recall"],
            "group": agg,
            "per_class": per_class,
            "val_gt_total": int(np.sum(nt_per_class)) if nt_per_class is not None else 0,
        }
        (arm / "metrics.json").write_text(json.dumps(payload, indent=2))

        summary[arm.name] = {
            "checkpoint": provenance["checkpoint"],
            "imgsz": provenance["imgsz"],
            **{k: v for k, v in payload.items()
               if k in ("AP", "AP50", "precision", "recall",
                        "head_AP", "head_AP50", "head_precision", "head_recall",
                        "tail_AP", "tail_AP50", "tail_precision", "tail_recall",
                        "val_gt_total")},
            "group": agg,
            "per_class_AP50": [per_class[NAMES[c]]["AP50"] for c in range(NC)],
            "per_class_AP": [per_class[NAMES[c]]["AP"] for c in range(NC)],
        }
        print(f"[eval] {arm.name}: mAP50={agg['overall']['mAP50']:.4f}  "
              f"mAP50-95={agg['overall']['mAP50-95']:.4f}  "
              f"head_mAP50={agg['head']['mAP50']:.4f}  tail_mAP50={agg['tail']['mAP50']:.4f}  "
              f"P={agg['overall']['Precision']:.4f}  R={agg['overall']['Recall']:.4f}", flush=True)

    out = runs_root / "gtsdb_eval_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"[eval] wrote {out}")

    ranked = sorted(((v.get("tail_AP50", 0.0), k) for k, v in summary.items()
                     if isinstance(v, dict) and "tail_AP50" in v), reverse=True)
    print("[eval] tail mAP50 ranking:")
    for i, (score, name) in enumerate(ranked, 1):
        print(f"  {i:2d}. {name:<28} {score:.4f}")

    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="datasets", type=Path)
    ap.add_argument("--data", default=None, type=Path,
                    help="dataset yaml (default: <data-root>/gtsdb_yolo.yaml)")
    ap.add_argument("--runs-root", default="runs", type=Path,
                    help="directory holding <arm>/best.pt run dirs")
    ap.add_argument("--device", default="0")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--only", default=None, help="comma-separated arm dir names")
    ap.add_argument("--no-plots", action="store_true",
                    help="skip the native Ultralytics val plots (runs/<arm>/eval_plots/)")
    args = ap.parse_args()
    evaluate(**vars(args))


if __name__ == "__main__":
    main()
