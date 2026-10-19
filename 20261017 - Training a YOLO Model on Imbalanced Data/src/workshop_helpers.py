"""Shared helpers for the GTSDB class-imbalance workshop notebook.
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HEAD_COLOR = "#2ecc71"
TAIL_COLOR = "#e74c3c"

METRICS_SCHEMA = 2  


def create_helpers(
    *,
    runs: Path,
    data_root: Path,
) -> dict[str, Callable]:

    def load_metrics(arm: str) -> dict:
        with open(runs / arm / "metrics.json", encoding="utf-8") as f:
            return json.load(f)

    def has_weights(arm: str) -> bool:
        """True when this arm already has a checkpoint (so training can be skipped)."""
        return (runs / arm / "best.pt").exists()

    def has_eval(arm: str) -> bool:
        """True when this arm already has evaluation metrics in the current schema.
        """
        path = runs / arm / "metrics.json"
        if not path.exists():
            return False
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f).get("schema") == METRICS_SCHEMA
        except (json.JSONDecodeError, OSError):
            return False

    def finish_run(arm: str, save_dir: str | Path) -> None:
        """Copy an Ultralytics run's outputs into runs/<arm>/ (best, last, results.csv)."""
        out_dir = runs / arm
        out_dir.mkdir(parents=True, exist_ok=True)
        run_dir = Path(save_dir)
        for filename in ("best.pt", "last.pt"):
            source = run_dir / "weights" / filename
            if source.exists() and source.resolve() != (out_dir / filename).resolve():
                shutil.copy(source, out_dir / filename)
        results_csv = run_dir / "results.csv"
        if results_csv.exists() and results_csv.resolve() != (out_dir / "results.csv").resolve():
            shutil.copy(results_csv, out_dir / "results.csv")

    def evaluate_one(arm: str, imgsz: int = 1280, device: str = "0") -> None:
        """Evaluate one arm with the Ultralytics validator, or load existing metrics."""
        import gtsdb_eval as evaluation

        if not has_eval(arm):
            tag = "EVAL"
            print("[EVAL] %s - running the Ultralytics validator" % arm)
            evaluation.evaluate(
                data_root=data_root,
                runs_root=runs,
                imgsz=imgsz,
                device=device,
                only=arm,
            )
        else:
            tag = "LOAD"

        metrics = load_metrics(arm)
        print(
            "[%s] %s: mAP50=%.3f tail_mAP50=%.3f head_mAP50=%.3f recall=%.3f"
            % (
                tag, arm,
                metrics.get("AP50", -1), metrics.get("tail_AP50", -1),
                metrics.get("head_AP50", -1), metrics.get("recall", -1),
            )
        )

    return {
        "load_metrics": load_metrics,
        "has_weights": has_weights,
        "has_eval": has_eval,
        "finish_run": finish_run,
        "evaluate_one": evaluate_one,
    }


# --------------------------------------------------------------------------- #
# Comparison / reporting helpers
# --------------------------------------------------------------------------- #

def metrics_table(arms, runs: Path, head_min: int | None = None) -> pd.DataFrame:
    """One wide row per arm: overall / head / tail mAP50, mAP50-95, Precision, Recall."""
    rows = {}
    for arm in arms:
        with open(runs / arm / "metrics.json", encoding="utf-8") as f:
            m = json.load(f)
        rows[arm] = {
            "mAP50": m.get("AP50"),
            "mAP50-95": m.get("AP"),
            "Precision": m.get("precision"),
            "Recall": m.get("recall"),
            "head_mAP50": m.get("head_AP50"),
            "head_Precision": m.get("head_precision"),
            "head_Recall": m.get("head_recall"),
            "tail_mAP50": m.get("tail_AP50"),
            "tail_Precision": m.get("tail_precision"),
            "tail_Recall": m.get("tail_recall"),
        }
    return pd.DataFrame(rows).T


def group_summary(arm: str, runs: Path) -> pd.DataFrame:
    """Rows overall/head/tail x columns mAP50, mAP50-95, Precision, Recall for one arm."""
    with open(runs / arm / "metrics.json", encoding="utf-8") as f:
        m = json.load(f)
    rows = {
        "overall": [m.get("AP50"), m.get("AP"), m.get("precision"), m.get("recall")],
        "head": [m.get("head_AP50"), m.get("head_AP"), m.get("head_precision"), m.get("head_recall")],
        "tail": [m.get("tail_AP50"), m.get("tail_AP"), m.get("tail_precision"), m.get("tail_recall")],
    }
    return pd.DataFrame(rows, index=["mAP50", "mAP50-95", "Precision", "Recall"]).T


def per_class_map_chart(arm: str, runs: Path, class_names: list[str], metric: str = "AP50"):
    """Per-class AP50 bars for one arm, coloured and ordered head vs tail."""
    with open(runs / arm / "metrics.json", encoding="utf-8") as f:
        m = json.load(f)
    per_class = m["per_class"]
    items = [(name, per_class[name].get(metric), per_class[name]["n_train"], per_class[name]["group"])
             for name in class_names]
    items = [it for it in items if it[1] is not None]
    items.sort(key=lambda t: t[1], reverse=True)
    names = [f"{t[0]} (n={t[2]})" for t in items]
    values = [t[1] for t in items]
    colors = [HEAD_COLOR if t[3] == "head" else TAIL_COLOR for t in items]

    fig, ax = plt.subplots(figsize=(6, 8.5))
    ax.barh(range(len(items)), values, color=colors)
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels(names, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel(metric)
    ax.set_title(f"{arm} - per-class {metric} (green = head, red = tail)")
    plt.tight_layout()
    plt.show()


def compare_arms_bars(arms, runs: Path, metric_keys=("mAP50", "head_mAP50", "tail_mAP50")):
    """Grouped bars per arm across the requested metric keys."""
    labels = {"mAP50": "overall", "head_mAP50": "head", "tail_mAP50": "tail",
              "Precision": "overall", "head_Precision": "head", "tail_Precision": "tail",
              "Recall": "overall", "head_Recall": "head", "tail_Recall": "tail"}
    df = metrics_table(arms, runs)[list(metric_keys)]
    ax = df.plot(kind="bar", figsize=(max(5.0, 1.05 * len(arms)), 3.0), width=0.8)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("score")
    ax.set_title("Comparison - " + ", ".join(labels.get(k, k) for k in metric_keys))
    ax.legend(loc="upper right", fontsize=8)
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.show()
    return df


def training_overview(arms, runs: Path, title: str = "Training overview"):
    """2 panels: overall loss (train vs val) and val mAP50 over epochs, one colour per arm.
    """
    arms = [a for a in arms if (Path(runs) / a / "results.csv").exists()]
    if not arms:
        print("no results.csv found - train an arm first")
        return
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.4))
    cmap = plt.get_cmap("tab10")
    train_keys = ("train/box_loss", "train/cls_loss", "train/dfl_loss")
    val_keys = ("val/box_loss", "val/cls_loss", "val/dfl_loss")
    for i, arm in enumerate(arms):
        with open(Path(runs) / arm / "results.csv", newline="") as f:
            rows = list(csv.DictReader(f))
        ep = [int(float(r["epoch"])) for r in rows]
        train = [sum(float(r[k]) for k in train_keys) for r in rows]
        val = [sum(float(r[k]) for k in val_keys) for r in rows]
        color = cmap(i % 10)
        axes[0].plot(ep, train, color=color, ls="-", lw=1.6, label="%s train" % arm)
        axes[0].plot(ep, val, color=color, ls="--", lw=1.4, label="%s val" % arm)
        axes[1].plot(ep, [float(r["metrics/mAP50(B)"]) for r in rows], color=color, lw=1.6, label=arm)
    axes[0].set_title("overall loss (train solid / val dashed)", fontsize=10)
    axes[1].set_title("val mAP50", fontsize=10)
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.show()


# --------------------------------------------------------------------------- #
# Per-arm visual analytics 
# --------------------------------------------------------------------------- #

GROUPS = ("overall", "head", "tail")
GROUP_METRICS = ("mAP50", "mAP50-95", "Precision", "Recall")


def _read_metrics(arm: str, runs: Path):
    path = Path(runs) / arm / "metrics.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _read_results(arm: str, runs: Path):
    path = Path(runs) / arm / "results.csv"
    if not path.exists():
        return None
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return rows or None


def _fc(rows, key):
    return [float(r[key]) for r in rows]


def _not_evaluated(arm: str):
    print("[%s: not evaluated yet - run/evaluate first]" % arm)


def group_delta_table(arm: str, runs: Path, baseline: str = "baseline"):
    """overall/head/tail mAP50, mAP50-95, Precision, Recall for baseline vs arm (+ delta)."""
    base = _read_metrics(baseline, runs)
    cur = _read_metrics(arm, runs)
    if base is None or cur is None:
        _not_evaluated(arm)
        return None
    cols = {}
    for met in GROUP_METRICS:
        b = [base["group"][bk].get(met) for bk in GROUPS]
        c = [cur["group"][bk].get(met) for bk in GROUPS]
        cols[(met, "baseline")] = b
        cols[(met, "arm")] = c
        cols[(met, "delta")] = [ci - bi for ci, bi in zip(c, b)]
    df = pd.DataFrame(cols, index=list(GROUPS))
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["metric", ""])
    return df


def _per_class_rows(arm, runs, class_names, baseline, metric):
    base = _read_metrics(baseline, runs)
    cur = _read_metrics(arm, runs)
    if base is None or cur is None:
        return None
    rows = {}
    for name in class_names:
        c = cur["per_class"].get(name)
        if not c or c.get(metric) is None:
            continue
        b = (base["per_class"].get(name) or {}) if base else {}
        bv = b.get(metric)
        av = c.get(metric)
        rows[name] = {
            "group": c.get("group"),
            "n_train": c.get("n_train"),
            "baseline": bv,
            "arm": av,
            "delta": (av - bv) if bv is not None else float("nan"),
        }
    return rows


def per_class_delta_table(arm: str, runs: Path, class_names: list[str],
                          baseline: str = "baseline", metric: str = "AP50"):
    """Per-class metric for baseline vs arm (+ delta), sorted by delta (best first)."""
    rows = _per_class_rows(arm, runs, class_names, baseline, metric)
    if rows is None:
        _not_evaluated(arm)
        return None
    df = pd.DataFrame(rows).T
    return df.sort_values("delta", ascending=False, na_position="last")


def per_class_delta_chart(arm: str, runs: Path, class_names: list[str],
                          baseline: str = "baseline", metric: str = "AP50"):
    """Two panels: per-class baseline-vs-arm bars, and the signed delta per class."""
    rows = _per_class_rows(arm, runs, class_names, baseline, metric)
    if rows is None:
        _not_evaluated(arm)
        return
    names = list(rows.keys())
    names.sort(key=lambda n: (rows[n]["delta"] if rows[n]["delta"] == rows[n]["delta"] else -9), reverse=True)
    base_v = [rows[n]["baseline"] if rows[n]["baseline"] is not None else float("nan") for n in names]
    arm_v = [rows[n]["arm"] for n in names]
    delta = [rows[n]["delta"] for n in names]
    colors = [HEAD_COLOR if rows[n]["group"] == "head" else TAIL_COLOR for n in names]

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, max(4.2, 0.22 * len(names))), sharey=True)
    y = np.arange(len(names))
    ax0.barh(y - 0.2, base_v, height=0.4, color="#95a5a6", label=baseline)
    ax0.barh(y + 0.2, arm_v, height=0.4, color=colors, label=arm)
    ax0.set_yticks(y)
    ax0.set_yticklabels(["%s (n=%s)" % (n, rows[n]["n_train"]) for n in names], fontsize=7)
    ax0.invert_yaxis()
    ax0.set_xlabel(metric)
    ax0.set_title("per-class %s: %s vs %s" % (metric, baseline, arm))
    ax0.legend(fontsize=8)

    ax1.barh(y, delta, color=colors)
    ax1.axvline(0, color="k", lw=0.8)
    ax1.set_xlabel("delta %s (%s - %s)" % (metric, arm, baseline))
    ax1.set_title("change vs baseline (green = head, red = tail)")
    plt.tight_layout()
    plt.show()


def metric_matrix(arms, runs: Path, metric: str = "mAP50") -> pd.DataFrame:
    """Wide table: one row per arm, columns overall / head / tail for one metric."""
    rows = {}
    for arm in arms:
        m = _read_metrics(arm, runs)
        if m is None:
            continue
        rows[arm] = {bk: m["group"][bk].get(metric) for bk in GROUPS}
    return pd.DataFrame(rows).T


def group_heatmap(arms, runs: Path, metric: str = "mAP50"):
    """Heatmap of overall / head / tail metric across arms."""
    df = metric_matrix(arms, runs, metric=metric)
    if df.empty:
        print("[no evaluated arms]")
        return
    fig, ax = plt.subplots(figsize=(4.2, max(2.1, 0.38 * len(df))))
    im = ax.imshow(df.values.astype(float), cmap="viridis", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(df.columns)))
    ax.set_xticklabels(df.columns)
    ax.set_yticks(range(len(df.index)))
    ax.set_yticklabels(df.index)
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            v = df.values[i, j]
            if v == v:
                ax.text(j, i, "%.2f" % v, ha="center", va="center",
                        color="white" if v < 0.55 else "black", fontsize=8)
    ax.set_title("group %s" % metric)
    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.show()


def per_class_heatmap(arms, runs: Path, class_names: list[str], metric: str = "AP50",
                      baseline: str | None = None, plot: bool = True):
    """Heatmap of per-class metric: arms x classes. If baseline is given, shows delta vs it."""
    data = {}
    for arm in arms:
        m = _read_metrics(arm, runs)
        if m is None:
            continue
        data[arm] = {name: (m["per_class"].get(name) or {}).get(metric) for name in class_names}
    df = pd.DataFrame(data).T
    if df.empty:
        print("[no evaluated arms]")
        return df
    delta = baseline is not None and baseline in df.index
    if delta:
        df = df.subtract(df.loc[baseline], axis=1)
    vmax = float(np.nanmax(np.abs(df.values))) if delta else 1.0
    if plot:
        fig, ax = plt.subplots(figsize=(max(7.5, 0.21 * len(class_names)), max(2.2, 0.4 * len(df))))
        im = ax.imshow(df.values.astype(float), cmap="coolwarm" if delta else "viridis",
                       aspect="auto", vmin=(-vmax if delta else 0), vmax=vmax)
        ax.set_xticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=90, fontsize=6)
        ax.set_yticks(range(len(df.index)))
        ax.set_yticklabels(df.index)
        title = "per-class %s" % metric
        if delta:
            title += " (delta vs %s)" % baseline
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.7)
        plt.tight_layout()
        plt.show()
    return df


def rank_arms(arms, runs: Path, metric: str = "tail_AP50") -> pd.DataFrame:
    """Arms ranked by a top-level metric (default tail mAP50)."""
    scores = [(_m.get(metric), arm) for arm in arms
              for _m in [_read_metrics(arm, runs)] if _m is not None and _m.get(metric) is not None]
    df = pd.DataFrame(scores, columns=[metric, "arm"]).sort_values(metric, ascending=False)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df

