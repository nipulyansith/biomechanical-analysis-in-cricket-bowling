"""
analyse_iav_better.py
=====================
Cleaner Inter-Annotator Variability analysis for manual keypoint annotation.

Main metric:
    - MPJPE (Mean Per Joint Position Error)

Computed in:
    - pixels
    - centimeters (from x_meter / y_meter if available)

Outputs:
    - iav_summary_statistics.csv
    - iav_pairwise_distances.csv
    - iav_centroid_deviation.csv
    - iav_report.txt
    - plot_01_jointwise_mpjpe_px.png
    - plot_02_jointwise_mpjpe_cm.png
    - plot_03_jointwise_boxplot_px.png
    - plot_04_jointwise_boxplot_cm.png
    - plot_05_pairwise_heatmap_px.png
    - plot_06_temporal_variability_px.png
    - plot_07_temporal_variability_cm.png
    - plot_08_not_visible_rate.png
    - plot_09_experience_comparison_cm.png   (if multiple levels exist)

Usage:
    python analyse_iav_better.py
    python analyse_iav_better.py --dir annotations --out iav_results

Dependencies:
    pip install pandas numpy scipy matplotlib
Optional:
    pip install pingouin krippendorff
"""

import os
import sys
import glob
import argparse
import warnings
from itertools import combinations
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import pingouin as pg
    HAS_PINGOUIN = True
except ImportError:
    HAS_PINGOUIN = False

try:
    import krippendorff
    HAS_KRIPP = True
except ImportError:
    HAS_KRIPP = False


JOINT_ORDER = [
    "Head",
    "Left Shoulder",  "Right Shoulder",
    "Left Elbow",     "Right Elbow",
    "Left Wrist",     "Right Wrist",
    "Left Knee",      "Right Knee",
    "Left Ankle",     "Right Ankle",
]

JOINT_COLORS = {
    "Head": "#FF6B6B",
    "Left Shoulder": "#FF9F43",
    "Right Shoulder": "#FECA57",
    "Left Elbow": "#48DBFB",
    "Right Elbow": "#FF9FF3",
    "Left Wrist": "#54A0FF",
    "Right Wrist": "#5F27CD",
    "Left Knee": "#00D2D3",
    "Right Knee": "#1DD1A1",
    "Left Ankle": "#C8D6E5",
    "Right Ankle": "#EE5A24",
}

BG = "#0f1117"
AX_BG = "#161a22"
FG = "#e6edf3"
GRID = "#2a2f3a"
ACCENT = "#6ee7b7"
WARN = "#fbbf24"
DANGER = "#fb7185"
MUTED = "#8b949e"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": AX_BG,
    "axes.edgecolor": GRID,
    "axes.labelcolor": FG,
    "axes.titlecolor": FG,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "grid.color": GRID,
    "text.color": FG,
    "legend.facecolor": AX_BG,
    "legend.edgecolor": GRID,
    "font.size": 9,
    "figure.dpi": 140,
})


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def ordered_joint_list(joints):
    present = set(joints)
    ordered = [j for j in JOINT_ORDER if j in present]
    extras = [j for j in joints if j not in JOINT_ORDER]
    return ordered + sorted(extras)


def load_annotations(csv_dir: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))
    if not files:
        sys.exit(f"[ERROR] No CSV files found in: {csv_dir}")

    dfs = []
    for path in files:
        try:
            df = pd.read_csv(path)
            df["source_file"] = os.path.basename(path)
            dfs.append(df)
        except Exception as e:
            warnings.warn(f"Could not read {path}: {e}")

    if not dfs:
        sys.exit("[ERROR] No readable CSV files found.")

    data = pd.concat(dfs, ignore_index=True)

    required = ["annotator_name", "video_id", "frame_number", "joint_name", "x_pixel", "y_pixel"]
    missing = [c for c in required if c not in data.columns]
    if missing:
        sys.exit(f"[ERROR] Missing required columns: {missing}")

    # Normalize types
    for col in ["x_pixel", "y_pixel", "x_meter", "y_meter"]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    data["frame_number"] = pd.to_numeric(data["frame_number"], errors="coerce").astype("Int64")

    if "skipped" not in data.columns:
        data["skipped"] = False
    if "not_visible" not in data.columns:
        data["not_visible"] = False
    if "experience_level" not in data.columns:
        data["experience_level"] = "unknown"

    data["skipped"] = data["skipped"].astype(str).str.lower().isin(["true", "1", "yes"])
    data["not_visible"] = data["not_visible"].astype(str).str.lower().isin(["true", "1", "yes"])

    data["is_valid_px"] = (
        ~data["skipped"] &
        ~data["not_visible"] &
        data["x_pixel"].notna() &
        data["y_pixel"].notna()
    )

    has_metric = {"x_meter", "y_meter"}.issubset(data.columns)
    data["has_metric_coords"] = False
    data["x_cm"] = np.nan
    data["y_cm"] = np.nan
    data["is_valid_cm"] = False

    if has_metric:
        data["x_cm"] = data["x_meter"] * 100.0
        data["y_cm"] = data["y_meter"] * 100.0
        data["has_metric_coords"] = data["x_meter"].notna() & data["y_meter"].notna()
        data["is_valid_cm"] = data["is_valid_px"] & data["has_metric_coords"]

    print(f"\n[INFO] Loaded {len(files)} CSV file(s)")
    print(f"[INFO] Rows         : {len(data):,}")
    print(f"[INFO] Annotators   : {sorted(data['annotator_name'].dropna().unique().tolist())}")
    print(f"[INFO] Videos       : {sorted(data['video_id'].dropna().unique().tolist())}")
    print(f"[INFO] Frames       : {data['frame_number'].nunique()}")
    print(f"[INFO] Joints       : {sorted(data['joint_name'].dropna().unique().tolist())}")
    print(f"[INFO] Metric coords: {'yes' if has_metric else 'no'}")
    return data


def compute_pairwise_distances(data: pd.DataFrame, coord_space: str = "px") -> pd.DataFrame:
    """
    coord_space: 'px' or 'cm'
    Returns one row per (video, frame, joint, annotator pair)
    """
    if coord_space == "px":
        valid_mask = data["is_valid_px"]
        x_col, y_col = "x_pixel", "y_pixel"
        dist_col = "distance_px"
    elif coord_space == "cm":
        valid_mask = data["is_valid_cm"]
        x_col, y_col = "x_cm", "y_cm"
        dist_col = "distance_cm"
    else:
        raise ValueError("coord_space must be 'px' or 'cm'")

    valid = data[valid_mask].copy()
    records = []

    for (vid, frame, joint), grp in valid.groupby(["video_id", "frame_number", "joint_name"]):
        pts = {
            row["annotator_name"]: (float(row[x_col]), float(row[y_col]))
            for _, row in grp.iterrows()
        }
        if len(pts) < 2:
            continue

        for a1, a2 in combinations(sorted(pts.keys()), 2):
            d = euclidean(pts[a1], pts[a2])
            records.append({
                "video_id": vid,
                "frame_number": int(frame),
                "joint_name": joint,
                "annotator_1": a1,
                "annotator_2": a2,
                "pair": f"{a1} vs {a2}",
                dist_col: float(d),
            })

    return pd.DataFrame(records)


def compute_centroid_deviation(data: pd.DataFrame, coord_space: str = "px") -> pd.DataFrame:
    if coord_space == "px":
        valid_mask = data["is_valid_px"]
        x_col, y_col = "x_pixel", "y_pixel"
        dev_col = "deviation_px"
    elif coord_space == "cm":
        valid_mask = data["is_valid_cm"]
        x_col, y_col = "x_cm", "y_cm"
        dev_col = "deviation_cm"
    else:
        raise ValueError("coord_space must be 'px' or 'cm'")

    valid = data[valid_mask].copy()
    records = []

    for (vid, frame, joint), grp in valid.groupby(["video_id", "frame_number", "joint_name"]):
        cx = grp[x_col].mean()
        cy = grp[y_col].mean()
        for _, row in grp.iterrows():
            d = euclidean((row[x_col], row[y_col]), (cx, cy))
            records.append({
                "video_id": vid,
                "frame_number": int(frame),
                "joint_name": joint,
                "annotator_name": row["annotator_name"],
                "experience_level": row.get("experience_level", "unknown"),
                dev_col: float(d),
            })

    return pd.DataFrame(records)


def compute_not_visible_rate(data: pd.DataFrame) -> pd.DataFrame:
    total = data.groupby(["joint_name", "annotator_name"]).size().rename("total_count")
    nv = data[data["not_visible"]].groupby(["joint_name", "annotator_name"]).size().rename("not_visible_count")
    out = pd.concat([total, nv], axis=1).fillna(0)
    out["not_visible_rate"] = out["not_visible_count"] / out["total_count"]
    return out.reset_index()


def compute_icc(data: pd.DataFrame, coord_col: str) -> pd.DataFrame:
    """
    ICC(2,1) per joint for a single coordinate column.
    coord_col should be one of:
        x_pixel, y_pixel, x_cm, y_cm
    """
    if coord_col in ("x_pixel", "y_pixel"):
        valid = data[data["is_valid_px"]].copy()
    else:
        valid = data[data["is_valid_cm"]].copy()

    results = []

    for joint, grp in valid.groupby("joint_name"):
        pivot = grp.pivot_table(
            index=["video_id", "frame_number"],
            columns="annotator_name",
            values=coord_col,
            aggfunc="first"
        ).dropna()

        if pivot.shape[0] < 3 or pivot.shape[1] < 2:
            results.append({"joint_name": joint, "coord": coord_col, "icc": np.nan})
            continue

        if HAS_PINGOUIN:
            long = (
                pivot.reset_index(drop=True)
                .melt(ignore_index=False, var_name="rater", value_name="value")
                .reset_index()
                .rename(columns={"index": "target"})
            )
            try:
                icc_df = pg.intraclass_corr(
                    data=long, targets="target", raters="rater", ratings="value", nan_policy="omit"
                )
                row = icc_df[icc_df["Type"] == "ICC2"].iloc[0]
                results.append({"joint_name": joint, "coord": coord_col, "icc": float(row["ICC"])})
            except Exception:
                results.append({"joint_name": joint, "coord": coord_col, "icc": np.nan})
        else:
            results.append({"joint_name": joint, "coord": coord_col, "icc": np.nan})

    return pd.DataFrame(results)


def compute_krippendorff_alpha(data: pd.DataFrame, coord_col: str) -> pd.DataFrame:
    if not HAS_KRIPP:
        return pd.DataFrame(columns=["joint_name", "coord", "kripp_alpha"])

    if coord_col in ("x_pixel", "y_pixel"):
        valid = data[data["is_valid_px"]].copy()
    else:
        valid = data[data["is_valid_cm"]].copy()

    results = []
    annotators = sorted(valid["annotator_name"].dropna().unique().tolist())

    for joint, grp in valid.groupby("joint_name"):
        pivot = grp.pivot_table(
            index=["video_id", "frame_number"],
            columns="annotator_name",
            values=coord_col,
            aggfunc="first"
        )
        if pivot.shape[0] < 3 or pivot.shape[1] < 2:
            results.append({"joint_name": joint, "coord": coord_col, "kripp_alpha": np.nan})
            continue

        pivot = pivot.reindex(columns=annotators)
        try:
            alpha = krippendorff.alpha(
                reliability_data=pivot.T.values,
                level_of_measurement="interval"
            )
            results.append({"joint_name": joint, "coord": coord_col, "kripp_alpha": float(alpha)})
        except Exception:
            results.append({"joint_name": joint, "coord": coord_col, "kripp_alpha": np.nan})

    return pd.DataFrame(results)


def summarize_pairwise(pair_px: pd.DataFrame, pair_cm: pd.DataFrame,
                       dev_px: pd.DataFrame, dev_cm: pd.DataFrame,
                       nv_rate: pd.DataFrame,
                       icc_tables: dict, kripp_tables: dict) -> pd.DataFrame:
    joints = ordered_joint_list(
        set(pair_px.get("joint_name", pd.Series(dtype=str)).dropna().tolist()) |
        set(pair_cm.get("joint_name", pd.Series(dtype=str)).dropna().tolist()) |
        set(nv_rate.get("joint_name", pd.Series(dtype=str)).dropna().tolist())
    )

    summary = pd.DataFrame({"joint_name": joints})

    if not pair_px.empty:
        px_stats = pair_px.groupby("joint_name")["distance_px"].agg(
            mpjpe_px="mean",
            sd_px="std",
            median_px="median",
            iqr_px=lambda x: x.quantile(0.75) - x.quantile(0.25),
            p95_px=lambda x: x.quantile(0.95),
            n_pairs_px="count",
        ).reset_index()
        summary = summary.merge(px_stats, on="joint_name", how="left")

    if not pair_cm.empty:
        cm_stats = pair_cm.groupby("joint_name")["distance_cm"].agg(
            mpjpe_cm="mean",
            sd_cm="std",
            median_cm="median",
            iqr_cm=lambda x: x.quantile(0.75) - x.quantile(0.25),
            p95_cm=lambda x: x.quantile(0.95),
            n_pairs_cm="count",
        ).reset_index()
        summary = summary.merge(cm_stats, on="joint_name", how="left")

    if not dev_px.empty:
        dev_px_stats = dev_px.groupby("joint_name")["deviation_px"].mean().reset_index().rename(
            columns={"deviation_px": "mean_centroid_dev_px"}
        )
        summary = summary.merge(dev_px_stats, on="joint_name", how="left")

    if not dev_cm.empty:
        dev_cm_stats = dev_cm.groupby("joint_name")["deviation_cm"].mean().reset_index().rename(
            columns={"deviation_cm": "mean_centroid_dev_cm"}
        )
        summary = summary.merge(dev_cm_stats, on="joint_name", how="left")

    nv_avg = nv_rate.groupby("joint_name")["not_visible_rate"].mean().reset_index().rename(
        columns={"not_visible_rate": "mean_not_visible_rate"}
    )
    summary = summary.merge(nv_avg, on="joint_name", how="left")

    for key, table in icc_tables.items():
        if not table.empty:
            summary = summary.merge(
                table.rename(columns={"icc": f"icc_{key}"}).drop(columns=["coord"], errors="ignore"),
                on="joint_name", how="left"
            )

    for key, table in kripp_tables.items():
        if not table.empty:
            summary = summary.merge(
                table.rename(columns={"kripp_alpha": f"kripp_alpha_{key}"}).drop(columns=["coord"], errors="ignore"),
                on="joint_name", how="left"
            )

    for col in summary.select_dtypes(include="float").columns:
        summary[col] = summary[col].round(4)

    return summary


def write_text_report(out_dir: str, data: pd.DataFrame, summary: pd.DataFrame,
                      pair_px: pd.DataFrame, pair_cm: pd.DataFrame) -> None:
    path = os.path.join(out_dir, "iav_report.txt")

    annotators = sorted(data["annotator_name"].dropna().unique().tolist())
    videos = sorted(data["video_id"].dropna().unique().tolist())

    overall_mpjpe_px = float(pair_px["distance_px"].mean()) if not pair_px.empty else np.nan
    overall_mpjpe_cm = float(pair_cm["distance_cm"].mean()) if not pair_cm.empty else np.nan

    best_joint = None
    worst_joint = None
    if "mpjpe_cm" in summary.columns and summary["mpjpe_cm"].notna().any():
        s = summary.dropna(subset=["mpjpe_cm"])
        if not s.empty:
            best_joint = s.loc[s["mpjpe_cm"].idxmin(), "joint_name"]
            worst_joint = s.loc[s["mpjpe_cm"].idxmax(), "joint_name"]
    elif "mpjpe_px" in summary.columns and summary["mpjpe_px"].notna().any():
        s = summary.dropna(subset=["mpjpe_px"])
        if not s.empty:
            best_joint = s.loc[s["mpjpe_px"].idxmin(), "joint_name"]
            worst_joint = s.loc[s["mpjpe_px"].idxmax(), "joint_name"]

    lines = []
    lines.append("INTER-ANNOTATOR VARIABILITY REPORT")
    lines.append("=" * 70)
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"Annotators: {len(annotators)}")
    lines.append(f"Videos: {len(videos)}")
    lines.append(f"Rows loaded: {len(data):,}")
    lines.append("")

    lines.append("GLOBAL SUMMARY")
    lines.append("-" * 70)
    lines.append(f"Overall MPJPE (pixel): {overall_mpjpe_px:.3f}" if not np.isnan(overall_mpjpe_px) else "Overall MPJPE (pixel): N/A")
    lines.append(f"Overall MPJPE (cm):    {overall_mpjpe_cm:.3f}" if not np.isnan(overall_mpjpe_cm) else "Overall MPJPE (cm): N/A")
    if best_joint:
        lines.append(f"Lowest-variability joint:  {best_joint}")
    if worst_joint:
        lines.append(f"Highest-variability joint: {worst_joint}")
    lines.append("")

    lines.append("PER-JOINT SUMMARY")
    lines.append("-" * 70)
    cols_to_show = [c for c in [
        "joint_name", "mpjpe_px", "mpjpe_cm", "sd_px", "sd_cm",
        "median_px", "median_cm", "p95_px", "p95_cm", "mean_not_visible_rate"
    ] if c in summary.columns]
    lines.append(summary[cols_to_show].to_string(index=False))
    lines.append("")

    lines.append("INTERPRETATION")
    lines.append("-" * 70)
    lines.append("Primary metric for this experiment is MPJPE.")
    lines.append("MPJPE in cm is the most interpretable real-world measure of disagreement.")
    lines.append("Frames/joints marked skipped or not_visible were excluded from distance calculations.")
    lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def save_plot(fig, out_dir: str, filename: str) -> None:
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, filename), bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def plot_jointwise_bar(summary: pd.DataFrame, value_col: str, ylabel: str, title: str, out_dir: str, filename: str):
    if value_col not in summary.columns or summary[value_col].dropna().empty:
        return

    joints = ordered_joint_list(summary["joint_name"].tolist())
    sub = summary.set_index("joint_name").loc[joints]
    vals = sub[value_col]

    fig, ax = plt.subplots(figsize=(11, 5))
    colors = [JOINT_COLORS.get(j, ACCENT) for j in joints]
    ax.bar(joints, vals, color=colors, alpha=0.9)
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.set_xticklabels(joints, rotation=40, ha="right")
    ax.grid(axis="y", alpha=0.35)
    save_plot(fig, out_dir, filename)


def plot_jointwise_box(pair_df: pd.DataFrame, value_col: str, ylabel: str, title: str, out_dir: str, filename: str):
    if pair_df.empty or value_col not in pair_df.columns:
        return

    joints = ordered_joint_list(pair_df["joint_name"].dropna().unique().tolist())
    data_list = [pair_df[pair_df["joint_name"] == j][value_col].dropna().values for j in joints]

    fig, ax = plt.subplots(figsize=(12, 5))
    bp = ax.boxplot(data_list, patch_artist=True, showfliers=True)
    for patch, j in zip(bp["boxes"], joints):
        patch.set_facecolor(JOINT_COLORS.get(j, ACCENT))
        patch.set_alpha(0.75)

    ax.set_xticks(range(1, len(joints) + 1))
    ax.set_xticklabels(joints, rotation=40, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.grid(axis="y", alpha=0.35)
    save_plot(fig, out_dir, filename)


def plot_pairwise_heatmap(pair_df: pd.DataFrame, out_dir: str):
    if pair_df.empty:
        return

    pivot = pair_df.groupby(["joint_name", "pair"])["distance_px"].mean().unstack()
    joints = [j for j in ordered_joint_list(pivot.index.tolist()) if j in pivot.index]
    pivot = pivot.loc[joints]

    fig, ax = plt.subplots(figsize=(max(8, 1.6 * len(pivot.columns)), 6))
    im = ax.imshow(pivot.values, aspect="auto", cmap="magma")

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(joints)))
    ax.set_yticklabels(joints)
    ax.set_title("Mean Pairwise Distance per Joint × Annotator Pair (px)", fontweight="bold")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Distance (px)")

    save_plot(fig, out_dir, "plot_05_pairwise_heatmap_px.png")


def plot_temporal_variability(pair_df: pd.DataFrame, value_col: str, ylabel: str, title: str, out_dir: str, filename: str):
    if pair_df.empty or value_col not in pair_df.columns:
        return

    frame_mean = pair_df.groupby("frame_number")[value_col].mean().reset_index().sort_values("frame_number")
    if frame_mean.empty:
        return

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(frame_mean["frame_number"], frame_mean[value_col], color=ACCENT, lw=1.8)
    ax.fill_between(frame_mean["frame_number"], frame_mean[value_col], alpha=0.15, color=ACCENT)

    if len(frame_mean) >= 3:
        slope, intercept, r, p, _ = stats.linregress(frame_mean["frame_number"], frame_mean[value_col])
        xs = frame_mean["frame_number"].values
        ax.plot(xs, intercept + slope * xs, color=WARN, ls="--", lw=1.2,
                label=f"trend slope={slope:.4f}, p={p:.4f}")
        ax.legend(fontsize=8)

    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Frame number")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.35)
    save_plot(fig, out_dir, filename)


def plot_not_visible_rate(nv_rate: pd.DataFrame, out_dir: str):
    if nv_rate.empty:
        return

    avg = nv_rate.groupby("joint_name")["not_visible_rate"].mean().reset_index()
    joints = ordered_joint_list(avg["joint_name"].tolist())
    avg = avg.set_index("joint_name").loc[joints].reset_index()

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.bar(avg["joint_name"], avg["not_visible_rate"] * 100.0,
           color=[JOINT_COLORS.get(j, WARN) for j in avg["joint_name"]], alpha=0.85)
    ax.set_title("Mean Not-Visible Rate per Joint", fontweight="bold")
    ax.set_ylabel("Not-visible rate (%)")
    ax.set_xticklabels(avg["joint_name"], rotation=40, ha="right")
    ax.grid(axis="y", alpha=0.35)
    save_plot(fig, out_dir, "plot_08_not_visible_rate.png")


def plot_experience_comparison(dev_cm: pd.DataFrame, out_dir: str):
    if dev_cm.empty or dev_cm["experience_level"].nunique() < 2:
        return

    joints = ordered_joint_list(dev_cm["joint_name"].dropna().unique().tolist())
    levels = sorted(dev_cm["experience_level"].dropna().unique().tolist())

    colors = {
        "novice": DANGER,
        "intermediate": WARN,
        "expert": ACCENT,
    }

    x = np.arange(len(joints))
    width = 0.8 / len(levels)

    fig, ax = plt.subplots(figsize=(12, 5))

    for i, level in enumerate(levels):
        sub = dev_cm[dev_cm["experience_level"] == level]
        means = [sub[sub["joint_name"] == j]["deviation_cm"].mean() for j in joints]
        pos = x + (i - (len(levels) - 1) / 2.0) * width
        ax.bar(pos, means, width=width, label=level, color=colors.get(level, "#7d8590"), alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(joints, rotation=40, ha="right")
    ax.set_ylabel("Mean deviation from centroid (cm)")
    ax.set_title("Experience-Level Comparison of Annotation Variability", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.35)
    save_plot(fig, out_dir, "plot_09_experience_comparison_cm.png")


def main():
    parser = argparse.ArgumentParser(description="Better Inter-Annotator Variability analysis")
    parser.add_argument("--dir", default="annotations", help="Input folder containing annotation CSVs")
    parser.add_argument("--out", default="iav_results", help="Output folder")
    args = parser.parse_args()

    ensure_dir(args.out)

    data = load_annotations(args.dir)

    print("[INFO] Computing pairwise distances...")
    pair_px = compute_pairwise_distances(data, "px")
    pair_cm = compute_pairwise_distances(data, "cm")

    if pair_px.empty and pair_cm.empty:
        sys.exit("[ERROR] No comparable annotator pairs found.")

    print("[INFO] Computing centroid deviations...")
    dev_px = compute_centroid_deviation(data, "px")
    dev_cm = compute_centroid_deviation(data, "cm")

    print("[INFO] Computing not-visible rates...")
    nv_rate = compute_not_visible_rate(data)

    print("[INFO] Computing ICC...")
    icc_tables = {
        "x_px": compute_icc(data, "x_pixel"),
        "y_px": compute_icc(data, "y_pixel"),
        "x_cm": compute_icc(data, "x_cm") if data["is_valid_cm"].any() else pd.DataFrame(),
        "y_cm": compute_icc(data, "y_cm") if data["is_valid_cm"].any() else pd.DataFrame(),
    }

    print("[INFO] Computing Krippendorff alpha...")
    kripp_tables = {
        "x_px": compute_krippendorff_alpha(data, "x_pixel"),
        "y_px": compute_krippendorff_alpha(data, "y_pixel"),
        "x_cm": compute_krippendorff_alpha(data, "x_cm") if data["is_valid_cm"].any() else pd.DataFrame(),
        "y_cm": compute_krippendorff_alpha(data, "y_cm") if data["is_valid_cm"].any() else pd.DataFrame(),
    }

    print("[INFO] Building summary...")
    summary = summarize_pairwise(pair_px, pair_cm, dev_px, dev_cm, nv_rate, icc_tables, kripp_tables)

    # Save tables
    summary.to_csv(os.path.join(args.out, "iav_summary_statistics.csv"), index=False)
    if not pair_px.empty:
        pair_px.to_csv(os.path.join(args.out, "iav_pairwise_distances_px.csv"), index=False)
    if not pair_cm.empty:
        pair_cm.to_csv(os.path.join(args.out, "iav_pairwise_distances_cm.csv"), index=False)
    if not dev_px.empty:
        dev_px.to_csv(os.path.join(args.out, "iav_centroid_deviation_px.csv"), index=False)
    if not dev_cm.empty:
        dev_cm.to_csv(os.path.join(args.out, "iav_centroid_deviation_cm.csv"), index=False)

    print("[INFO] Writing report...")
    write_text_report(args.out, data, summary, pair_px, pair_cm)

    print("[INFO] Generating plots...")
    plot_jointwise_bar(
        summary, "mpjpe_px",
        "MPJPE (pixels)",
        "Joint-wise MPJPE in Pixel Space",
        args.out, "plot_01_jointwise_mpjpe_px.png"
    )
    plot_jointwise_bar(
        summary, "mpjpe_cm",
        "MPJPE (cm)",
        "Joint-wise MPJPE in Real-World Space",
        args.out, "plot_02_jointwise_mpjpe_cm.png"
    )
    plot_jointwise_box(
        pair_px, "distance_px",
        "Pairwise distance (px)",
        "Distribution of Pairwise Distances per Joint (px)",
        args.out, "plot_03_jointwise_boxplot_px.png"
    )
    plot_jointwise_box(
        pair_cm, "distance_cm",
        "Pairwise distance (cm)",
        "Distribution of Pairwise Distances per Joint (cm)",
        args.out, "plot_04_jointwise_boxplot_cm.png"
    )
    plot_pairwise_heatmap(pair_px, args.out)
    plot_temporal_variability(
        pair_px, "distance_px",
        "Mean pairwise distance (px)",
        "Temporal Trend of Inter-Annotator Variability (px)",
        args.out, "plot_06_temporal_variability_px.png"
    )
    plot_temporal_variability(
        pair_cm, "distance_cm",
        "Mean pairwise distance (cm)",
        "Temporal Trend of Inter-Annotator Variability (cm)",
        args.out, "plot_07_temporal_variability_cm.png"
    )
    plot_not_visible_rate(nv_rate, args.out)
    plot_experience_comparison(dev_cm, args.out)

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Output folder: {os.path.abspath(args.out)}")
    if not pair_px.empty:
        print(f"Overall MPJPE (px): {pair_px['distance_px'].mean():.4f}")
    if not pair_cm.empty:
        print(f"Overall MPJPE (cm): {pair_cm['distance_cm'].mean():.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()