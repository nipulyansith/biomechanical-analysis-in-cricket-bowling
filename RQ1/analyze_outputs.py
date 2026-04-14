"""
=============================================================================
Pose Estimation Accuracy Analysis — RQ1
=============================================================================
Compares YOLOv8l-Pose and MediaPipe Pose against Manual Ground Truth.

Outputs per trial (and aggregated across all trials):
  1. accuracy_metrics.csv        — MEE, RMSE, Bias (X/Y) per joint × model
  2. reliability_metrics.csv     — Limb-length StdDev per model
  3. error_bar_chart.png         — Average error per joint, both models
  4. precision_scatter.png       — Click distribution for a chosen joint
  5. bland_altman.png            — Agreement plots (X and Y axes)
  6. summary_report.png          — Combined dashboard figure

Usage
-----
  # Single trial folder (must contain *_Manual.csv, *_YOLO.csv, *_MediaPipe.csv)
  python analyse_pose.py --input /path/to/TRIAL_001

  # Batch — loop over all sub-folders
  python analyse_pose.py --input /path/to/trials_folder --batch

  # Override which joint is used for the precision scatter
  python analyse_pose.py --input /path/to/TRIAL_001 --scatter-joint "R-Ankle"

=============================================================================
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)

# ─────────────────────────────────────────────────────────────────────────────
# Global style
# ─────────────────────────────────────────────────────────────────────────────
PALETTE = {
    "YOLO":      "#FF6B35",
    "MediaPipe": "#00C9A7",
    "Manual":    "#4A90D9",
}
JOINT_ORDER = [
    "Head",
    "L-Shoulder", "R-Shoulder",
    "L-Elbow",    "R-Elbow",
    "L-Wrist",    "R-Wrist",
    "L-Knee",     "R-Knee",
    "L-Ankle",    "R-Ankle",
]
LIMB_PAIRS = [
    ("L-Shoulder", "L-Elbow",    "L-Upper-Arm"),
    ("R-Shoulder", "R-Elbow",    "R-Upper-Arm"),
    ("L-Elbow",    "L-Wrist",    "L-Forearm"),
    ("R-Elbow",    "R-Wrist",    "R-Forearm"),
    ("L-Shoulder", "R-Shoulder", "Shoulder-Width"),
    ("L-Shoulder", "L-Knee",     "L-Trunk"),
    ("R-Shoulder", "R-Knee",     "R-Trunk"),
    ("L-Knee",     "L-Ankle",    "L-Lower-Leg"),
    ("R-Knee",     "R-Ankle",    "R-Lower-Leg"),
]

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.35,
    "grid.linestyle":    "--",
    "figure.dpi":        150,
    "savefig.dpi":       150,
    "savefig.bbox":      "tight",
})


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

REQUIRED_COLS = {"Frame_Num", "Joint_Name", "Meter_X", "Meter_Y"}


def _validate(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}")
    df["Frame_Num"] = df["Frame_Num"].astype(int)
    df["Joint_Name"] = df["Joint_Name"].str.strip()
    df["Meter_X"] = pd.to_numeric(df["Meter_X"], errors="coerce")
    df["Meter_Y"] = pd.to_numeric(df["Meter_Y"], errors="coerce")
    return df


def load_trial(folder: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Locate Manual / YOLO / MediaPipe CSVs inside *folder*.
    Filenames are matched by case-insensitive substring search so that
    any naming convention (e.g. TRIAL_001_Manual.csv or manual.csv) works.
    """
    def _find(keyword: str) -> Path:
        matches = [p for p in folder.glob("*.csv")
                   if keyword.lower() in p.stem.lower()]
        if not matches:
            raise FileNotFoundError(
                f"No CSV with '{keyword}' in its name found in {folder}")
        return matches[0]

    manual_path    = _find("manual")
    yolo_path      = _find("yolo")
    mediapipe_path = _find("mediapipe")

    manual    = _validate(pd.read_csv(manual_path),    manual_path)
    yolo      = _validate(pd.read_csv(yolo_path),      yolo_path)
    mediapipe = _validate(pd.read_csv(mediapipe_path), mediapipe_path)
    return manual, yolo, mediapipe


# ─────────────────────────────────────────────────────────────────────────────
# Coordinate normalisation
# ─────────────────────────────────────────────────────────────────────────────

def normalise_to_reference(
    df: pd.DataFrame,
    ref_df: pd.DataFrame,
    anchor_joint: str = "Head",
) -> pd.DataFrame:
    """
    Translates *df* so its frame-0 anchor joint aligns with *ref_df*'s.
    Useful when different calibration sessions produce different absolute
    origins.  A scale correction is also applied if the mean inter-shoulder
    distance differs by > 5 % between df and ref_df.
    """
    df = df.copy()

    def _get_origin(d: pd.DataFrame) -> tuple[float, float]:
        row = d[(d["Frame_Num"] == d["Frame_Num"].min()) &
                (d["Joint_Name"] == anchor_joint)]
        if row.empty:
            return 0.0, 0.0
        return float(row["Meter_X"].iloc[0]), float(row["Meter_Y"].iloc[0])

    ox_ref, oy_ref = _get_origin(ref_df)
    ox_df,  oy_df  = _get_origin(df)

    df["Meter_X"] = df["Meter_X"] + (ox_ref - ox_df)
    df["Meter_Y"] = df["Meter_Y"] + (oy_ref - oy_df)

    # Scale correction via inter-shoulder distance
    def _mean_shoulder_dist(d: pd.DataFrame) -> float | None:
        ls = d[d["Joint_Name"] == "L-Shoulder"][["Frame_Num","Meter_X","Meter_Y"]]
        rs = d[d["Joint_Name"] == "R-Shoulder"][["Frame_Num","Meter_X","Meter_Y"]]
        merged = ls.merge(rs, on="Frame_Num", suffixes=("_l","_r"))
        if merged.empty:
            return None
        dists = np.sqrt((merged["Meter_X_l"] - merged["Meter_X_r"])**2 +
                        (merged["Meter_Y_l"] - merged["Meter_Y_r"])**2)
        return float(dists.mean())

    sd_ref = _mean_shoulder_dist(ref_df)
    sd_df  = _mean_shoulder_dist(df)
    if sd_ref and sd_df and abs(sd_ref - sd_df) / sd_ref > 0.05:
        scale = sd_ref / sd_df
        cx = float(df["Meter_X"].mean())
        cy = float(df["Meter_Y"].mean())
        df["Meter_X"] = cx + (df["Meter_X"] - cx) * scale
        df["Meter_Y"] = cy + (df["Meter_Y"] - cy) * scale

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Core metrics
# ─────────────────────────────────────────────────────────────────────────────

def _merge_with_gt(
    gt: pd.DataFrame,
    model: pd.DataFrame,
    model_label: str,
) -> pd.DataFrame:
    """
    Inner-join GT and model on (Frame_Num, Joint_Name).
    Returns dataframe with columns: Joint_Name, err_x, err_y, euclidean.
    """
    merged = gt[["Frame_Num","Joint_Name","Meter_X","Meter_Y"]].merge(
        model[["Frame_Num","Joint_Name","Meter_X","Meter_Y"]],
        on=["Frame_Num","Joint_Name"],
        suffixes=("_gt","_model"),
    ).dropna(subset=["Meter_X_gt","Meter_Y_gt","Meter_X_model","Meter_Y_model"])

    merged["err_x"]      = merged["Meter_X_model"] - merged["Meter_X_gt"]
    merged["err_y"]      = merged["Meter_Y_model"] - merged["Meter_Y_gt"]
    merged["euclidean"]  = np.sqrt(merged["err_x"]**2 + merged["err_y"]**2)
    merged["model"]      = model_label
    return merged


def compute_accuracy_metrics(
    manual: pd.DataFrame,
    yolo: pd.DataFrame,
    mediapipe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Returns a tidy DataFrame with one row per (model × joint):
      Model | Joint_Name | MEE | RMSE | Bias_X | Bias_Y | N_frames
    """
    rows = []
    for label, df in [("YOLO", yolo), ("MediaPipe", mediapipe)]:
        merged = _merge_with_gt(manual, df, label)
        for joint in JOINT_ORDER:
            sub = merged[merged["Joint_Name"] == joint]
            if sub.empty:
                continue
            mee    = float(sub["euclidean"].mean())
            rmse   = float(np.sqrt((sub["euclidean"]**2).mean()))
            bias_x = float(sub["err_x"].mean())
            bias_y = float(sub["err_y"].mean())
            n      = len(sub)
            rows.append({
                "Model":      label,
                "Joint_Name": joint,
                "MEE_m":      round(mee,    6),
                "RMSE_m":     round(rmse,   6),
                "MEE_cm":     round(mee   * 100, 4),
                "RMSE_cm":    round(rmse  * 100, 4),
                "Bias_X_cm":  round(bias_x * 100, 4),
                "Bias_Y_cm":  round(bias_y * 100, 4),
                "N_Frames":   n,
            })
    return pd.DataFrame(rows)


def compute_reliability_metrics(
    manual: pd.DataFrame,
    yolo: pd.DataFrame,
    mediapipe: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each limb pair, compute per-frame limb length across all frames,
    then return StdDev (and mean) for each model.
    """
    rows = []
    for label, df in [("Manual", manual), ("YOLO", yolo), ("MediaPipe", mediapipe)]:
        for jA, jB, limb_name in LIMB_PAIRS:
            dfA = df[df["Joint_Name"] == jA][["Frame_Num","Meter_X","Meter_Y"]]
            dfB = df[df["Joint_Name"] == jB][["Frame_Num","Meter_X","Meter_Y"]]
            merged = dfA.merge(dfB, on="Frame_Num", suffixes=("_a","_b")).dropna()
            if merged.empty:
                continue
            lengths = np.sqrt(
                (merged["Meter_X_a"] - merged["Meter_X_b"])**2 +
                (merged["Meter_Y_a"] - merged["Meter_Y_b"])**2
            )
            rows.append({
                "Model":         label,
                "Limb":          limb_name,
                "Mean_Length_m": round(float(lengths.mean()),  6),
                "StdDev_m":      round(float(lengths.std()),   6),
                "CV_pct":        round(float(lengths.std() / lengths.mean() * 100) if lengths.mean() > 0 else np.nan, 3),
                "N_Frames":      len(lengths),
            })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Visualisations
# ─────────────────────────────────────────────────────────────────────────────

def plot_error_bar_chart(
    accuracy_df: pd.DataFrame,
    out_path: Path,
    trial_label: str = "",
) -> None:
    """
    Grouped bar chart: MEE (cm) per joint for YOLO vs MediaPipe.
    Error bars show ± RMSE.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False)
    metrics = [("MEE_cm", "RMSE_cm", "Mean Euclidean Error (cm)"),
               ("RMSE_cm", None,     "RMSE (cm)")]

    for ax, (metric_col, err_col, ylabel) in zip(axes, metrics):
        pivoted = accuracy_df.pivot(index="Joint_Name", columns="Model", values=metric_col)
        pivoted = pivoted.reindex([j for j in JOINT_ORDER if j in pivoted.index])

        x = np.arange(len(pivoted))
        width = 0.35
        models = ["YOLO", "MediaPipe"]

        for i, model in enumerate(models):
            if model not in pivoted.columns:
                continue
            vals = pivoted[model].values
            errs = None
            if err_col:
                err_pivot = accuracy_df.pivot(
                    index="Joint_Name", columns="Model", values=err_col)
                err_pivot = err_pivot.reindex(pivoted.index)
                errs = err_pivot[model].values if model in err_pivot.columns else None

            bars = ax.bar(
                x + i * width - width / 2, vals,
                width=width * 0.88,
                color=PALETTE[model],
                alpha=0.88,
                label=model,
                zorder=3,
            )
            if errs is not None:
                ax.errorbar(
                    x + i * width - width / 2, vals,
                    yerr=errs, fmt="none",
                    color="#333333", capsize=3, linewidth=1.2, zorder=4,
                )
            # Value labels on top of each bar
            for bar, v in zip(bars, vals):
                if not np.isnan(v):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01 * ax.get_ylim()[1],
                        f"{v:.2f}",
                        ha="center", va="bottom",
                        fontsize=7, color="#333333",
                    )

        ax.set_xticks(x)
        ax.set_xticklabels(pivoted.index, rotation=38, ha="right", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(ylabel, fontsize=11, fontweight="bold", pad=10)
        ax.legend(fontsize=9)

    title = "Per-Joint Error Metrics — YOLO vs MediaPipe"
    if trial_label:
        title += f"  [{trial_label}]"
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_precision_scatter(
    manual: pd.DataFrame,
    yolo: pd.DataFrame,
    mediapipe: pd.DataFrame,
    joint: str,
    out_path: Path,
    trial_label: str = "",
) -> None:
    """
    Scatter plot of AI-model coordinates for *joint*, expressed as
    displacement from the Manual ground-truth mean (treated as origin).
    Includes 95% confidence ellipses.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    gt_sub = manual[manual["Joint_Name"] == joint].dropna(
        subset=["Meter_X","Meter_Y"])
    gt_cx = gt_sub["Meter_X"].mean()
    gt_cy = gt_sub["Meter_Y"].mean()

    for ax, (label, df) in zip(axes, [("YOLO", yolo), ("MediaPipe", mediapipe)]):
        sub = df[df["Joint_Name"] == joint].dropna(subset=["Meter_X","Meter_Y"])
        dx = (sub["Meter_X"] - gt_cx) * 100   # → centimetres
        dy = (sub["Meter_Y"] - gt_cy) * 100

        color = PALETTE[label]
        ax.scatter(dx, dy, c=color, alpha=0.75, s=55, zorder=3,
                   edgecolors="white", linewidths=0.5, label=f"{label} clicks")

        # GT origin
        ax.scatter([0], [0], c=PALETTE["Manual"], s=140, zorder=5,
                   marker="*", label="GT origin")

        # 95 % confidence ellipse
        if len(dx) >= 3:
            _draw_ellipse(ax, dx.values, dy.values, color=color, n_std=2.0)

        # Mean marker
        ax.scatter([dx.mean()], [dy.mean()], c=color, s=90, zorder=6,
                   marker="D", edgecolors="black", linewidths=0.8,
                   label=f"{label} mean")

        # Statistics annotation
        mee = np.sqrt(dx**2 + dy**2).mean()
        textstr = (f"n = {len(dx)}\n"
                   f"MEE = {mee:.2f} cm\n"
                   f"σ_x = {dx.std():.2f} cm\n"
                   f"σ_y = {dy.std():.2f} cm")
        ax.text(0.97, 0.97, textstr, transform=ax.transAxes,
                fontsize=8, va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          alpha=0.85, edgecolor=color))

        ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
        ax.axvline(0, color="grey", linewidth=0.8, linestyle="--")
        ax.set_xlabel("X displacement (cm)", fontsize=10)
        ax.set_ylabel("Y displacement (cm)", fontsize=10)
        ax.set_title(f"{label}  —  {joint}", fontsize=11, fontweight="bold")
        ax.legend(fontsize=8, loc="lower right")
        ax.set_aspect("equal", adjustable="datalim")

    title = f"Precision Scatter — {joint}"
    if trial_label:
        title += f"  [{trial_label}]"
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _draw_ellipse(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    color: str,
    n_std: float = 2.0,
) -> None:
    """Draw a covariance confidence ellipse."""
    from matplotlib.patches import Ellipse
    import matplotlib.transforms as transforms

    cov = np.cov(x, y)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = eigenvalues.argsort()[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(np.abs(eigenvalues))

    ellipse = Ellipse(
        xy=(np.mean(x), np.mean(y)),
        width=width, height=height, angle=angle,
        edgecolor=color, facecolor=color,
        linewidth=1.8, linestyle="--", alpha=0.18, zorder=2,
    )
    ax.add_patch(ellipse)


def plot_bland_altman(
    manual: pd.DataFrame,
    yolo: pd.DataFrame,
    mediapipe: pd.DataFrame,
    out_path: Path,
    trial_label: str = "",
) -> None:
    """
    Four-panel Bland–Altman: (YOLO-X, YOLO-Y, MP-X, MP-Y)
    X-axis  = mean of model + manual  (cm)
    Y-axis  = model − manual          (cm)
    Limits of Agreement = mean ± 1.96 · SD
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    configs = [
        ("YOLO",      "X", yolo,      axes[0, 0]),
        ("YOLO",      "Y", yolo,      axes[0, 1]),
        ("MediaPipe", "X", mediapipe, axes[1, 0]),
        ("MediaPipe", "Y", mediapipe, axes[1, 1]),
    ]

    for model_label, axis_label, model_df, ax in configs:
        coord = f"Meter_{axis_label}"
        merged = manual[["Frame_Num","Joint_Name", coord]].merge(
            model_df[["Frame_Num","Joint_Name", coord]],
            on=["Frame_Num","Joint_Name"],
            suffixes=("_gt","_model"),
        ).dropna()

        gt_vals    = merged[f"{coord}_gt"].values    * 100   # cm
        model_vals = merged[f"{coord}_model"].values * 100

        mean_vals  = (gt_vals + model_vals) / 2
        diff_vals  = model_vals - gt_vals

        bias  = diff_vals.mean()
        sd    = diff_vals.std()
        loa_u = bias + 1.96 * sd
        loa_l = bias - 1.96 * sd

        # Colour by joint
        joint_ids = merged["Joint_Name"].map(
            {j: i for i, j in enumerate(JOINT_ORDER)})
        scatter = ax.scatter(
            mean_vals, diff_vals,
            c=joint_ids, cmap="tab20",
            alpha=0.65, s=22, zorder=3,
        )

        ax.axhline(bias,  color=PALETTE[model_label], linewidth=1.8,
                   linestyle="-",  label=f"Bias = {bias:.3f} cm")
        ax.axhline(loa_u, color=PALETTE[model_label], linewidth=1.3,
                   linestyle="--", label=f"+1.96 SD = {loa_u:.3f} cm")
        ax.axhline(loa_l, color=PALETTE[model_label], linewidth=1.3,
                   linestyle=":",  label=f"−1.96 SD = {loa_l:.3f} cm")
        ax.axhline(0, color="grey", linewidth=0.7, linestyle="-", alpha=0.5)

        # Shaded LoA band
        ax.axhspan(loa_l, loa_u, alpha=0.07,
                   color=PALETTE[model_label], zorder=1)

        # Annotate bias/LoA on right margin
        for y_val, lbl in [(bias, f"{bias:.3f}"),
                            (loa_u, f"{loa_u:.3f}"),
                            (loa_l, f"{loa_l:.3f}")]:
            ax.text(ax.get_xlim()[1] if ax.get_xlim()[1] != 0.0 else mean_vals.max(),
                    y_val, f" {lbl}",
                    va="center", fontsize=7.5, color=PALETTE[model_label])

        ax.set_xlabel(f"Mean of Manual & {model_label} — {axis_label} (cm)",
                      fontsize=9)
        ax.set_ylabel(f"{model_label} − Manual — {axis_label} (cm)", fontsize=9)
        ax.set_title(
            f"Bland–Altman: {model_label}  {axis_label}-axis\n"
            f"Bias={bias:.3f} cm  LoA=[{loa_l:.3f}, {loa_u:.3f}] cm",
            fontsize=10, fontweight="bold",
        )
        ax.legend(fontsize=7.5, loc="upper right")

        # Colorbar legend for joints
        cbar = fig.colorbar(scatter, ax=ax, pad=0.01)
        cbar.set_ticks(range(len(JOINT_ORDER)))
        cbar.set_ticklabels(JOINT_ORDER, fontsize=6)

    title = "Bland–Altman Agreement Analysis"
    if trial_label:
        title += f"  [{trial_label}]"
    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_summary_dashboard(
    accuracy_df: pd.DataFrame,
    reliability_df: pd.DataFrame,
    out_path: Path,
    trial_label: str = "",
) -> None:
    """
    Single-page dashboard combining:
      - Overall MEE heatmap (joint × model)
      - Bias quiver plot (direction of systematic drift)
      - Limb StdDev grouped bar chart
      - Summary statistics table
    """
    fig = plt.figure(figsize=(18, 14))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    ax_heat  = fig.add_subplot(gs[0, 0])
    ax_bias  = fig.add_subplot(gs[0, 1])
    ax_limb  = fig.add_subplot(gs[0, 2])
    ax_table = fig.add_subplot(gs[1, :])

    # ── 1. MEE heatmap ────────────────────────────────────────────────────
    heat_data = accuracy_df.pivot(
        index="Joint_Name", columns="Model", values="MEE_cm"
    ).reindex([j for j in JOINT_ORDER if j in accuracy_df["Joint_Name"].values])

    sns.heatmap(
        heat_data, ax=ax_heat, annot=True, fmt=".2f",
        cmap="YlOrRd", linewidths=0.4, cbar_kws={"label": "MEE (cm)"},
        annot_kws={"size": 9},
    )
    ax_heat.set_title("MEE Heatmap (cm)", fontweight="bold", fontsize=11)
    ax_heat.set_xlabel("")
    ax_heat.set_ylabel("")
    ax_heat.tick_params(axis="x", labelsize=9)
    ax_heat.tick_params(axis="y", labelsize=8)

    # ── 2. Bias quiver plot ───────────────────────────────────────────────
    for i, model in enumerate(["YOLO", "MediaPipe"]):
        sub = accuracy_df[accuracy_df["Model"] == model]
        joints_in = [j for j in JOINT_ORDER if j in sub["Joint_Name"].values]
        y_pos = np.arange(len(joints_in)) + i * 0.35 - 0.175

        for k, joint in enumerate(joints_in):
            row = sub[sub["Joint_Name"] == joint]
            if row.empty:
                continue
            bx = float(row["Bias_X_cm"].iloc[0])
            by = float(row["Bias_Y_cm"].iloc[0])
            ax_bias.quiver(
                0, y_pos[k], bx, 0,
                color=PALETTE[model], alpha=0.85,
                scale=1, scale_units="xy", angles="xy",
                width=0.008,
            )
            ax_bias.quiver(
                0, y_pos[k], 0, by,
                color=PALETTE[model], alpha=0.55,
                scale=1, scale_units="xy", angles="xy",
                width=0.005, linestyle="dashed",
            )

    ax_bias.axvline(0, color="grey", linewidth=0.8)
    ax_bias.set_yticks(np.arange(len(JOINT_ORDER)))
    ax_bias.set_yticklabels(JOINT_ORDER, fontsize=8)
    ax_bias.set_xlabel("Systematic Bias X (cm)", fontsize=9)
    ax_bias.set_title("System Bias Direction\n(solid=X, dashed=Y)", fontweight="bold", fontsize=11)
    patches = [mpatches.Patch(color=PALETTE[m], label=m) for m in ["YOLO","MediaPipe"]]
    ax_bias.legend(handles=patches, fontsize=8)

    # ── 3. Limb StdDev grouped bar chart ─────────────────────────────────
    limb_pivot = reliability_df.pivot(
        index="Limb", columns="Model", values="StdDev_m"
    ) * 100  # → cm
    limb_pivot = limb_pivot[[m for m in ["Manual","YOLO","MediaPipe"]
                              if m in limb_pivot.columns]]

    x = np.arange(len(limb_pivot))
    width = 0.25
    for i, model in enumerate(limb_pivot.columns):
        ax_limb.bar(
            x + i * width - width,
            limb_pivot[model].values,
            width=width * 0.9,
            color=PALETTE[model],
            alpha=0.85,
            label=model,
        )
    ax_limb.set_xticks(x)
    ax_limb.set_xticklabels(limb_pivot.index, rotation=38, ha="right", fontsize=7.5)
    ax_limb.set_ylabel("StdDev (cm)", fontsize=9)
    ax_limb.set_title("Limb Length Stability\n(lower = more stable)", fontweight="bold", fontsize=11)
    ax_limb.legend(fontsize=8)

    # ── 4. Summary table ──────────────────────────────────────────────────
    ax_table.axis("off")
    overall = (accuracy_df.groupby("Model")
               .agg(MEE_cm=("MEE_cm","mean"),
                    RMSE_cm=("RMSE_cm","mean"),
                    Bias_X_cm=("Bias_X_cm","mean"),
                    Bias_Y_cm=("Bias_Y_cm","mean"))
               .round(4)
               .reset_index())

    rel_summary = (reliability_df[reliability_df["Model"] != "Manual"]
                   .groupby("Model")
                   .agg(Mean_LimbStdDev_cm=("StdDev_m","mean"))
                   .apply(lambda x: x * 100)
                   .round(4)
                   .reset_index())

    summary = overall.merge(rel_summary, on="Model", how="left")
    summary.columns = ["Model","MEE (cm)","RMSE (cm)",
                        "Bias-X (cm)","Bias-Y (cm)","Mean Limb StdDev (cm)"]

    tbl = ax_table.table(
        cellText=summary.values,
        colLabels=summary.columns,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.2, 2.0)

    for j, col in enumerate(summary.columns):
        tbl[0, j].set_facecolor("#1a1a2e")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    for i in range(1, len(summary) + 1):
        model_name = summary.iloc[i - 1, 0]
        bg = PALETTE.get(model_name, "#f9f9f9") + "33"   # 20 % alpha hex
        for j in range(len(summary.columns)):
            tbl[i, j].set_facecolor(bg)

    title = "RQ1 Summary Dashboard — Pose Estimation Accuracy"
    if trial_label:
        title += f"  [{trial_label}]"
    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.01)
    fig.savefig(out_path)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation across trials
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_trials(
    accuracy_list: list[pd.DataFrame],
    reliability_list: list[pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Concatenate per-trial DataFrames and produce mean ± SD across trials.
    """
    acc_all = pd.concat(accuracy_list, ignore_index=True)
    rel_all = pd.concat(reliability_list, ignore_index=True)

    acc_agg = (acc_all.groupby(["Model","Joint_Name"])
               .agg(
                   MEE_cm_mean  =("MEE_cm",    "mean"),
                   MEE_cm_std   =("MEE_cm",    "std"),
                   RMSE_cm_mean =("RMSE_cm",   "mean"),
                   RMSE_cm_std  =("RMSE_cm",   "std"),
                   Bias_X_cm_mean=("Bias_X_cm","mean"),
                   Bias_Y_cm_mean=("Bias_Y_cm","mean"),
                   N_Trials     =("N_Frames",  "count"),
               )
               .round(6)
               .reset_index())

    rel_agg = (rel_all.groupby(["Model","Limb"])
               .agg(
                   StdDev_mean=("StdDev_m","mean"),
                   StdDev_sd  =("StdDev_m","std"),
                   CV_pct_mean=("CV_pct",  "mean"),
                   N_Trials   =("N_Frames","count"),
               )
               .round(6)
               .reset_index())

    return acc_agg, rel_agg


# ─────────────────────────────────────────────────────────────────────────────
# Per-trial runner
# ─────────────────────────────────────────────────────────────────────────────

def analyse_trial(
    trial_folder: Path,
    output_root: Path,
    scatter_joint: str = "R-Ankle",
    normalise: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full analysis pipeline for a single trial.
    Returns (accuracy_df, reliability_df).
    """
    trial_label = trial_folder.name
    out_dir = output_root / trial_label
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Trial: {trial_label}")
    print(f"{'='*60}")

    manual, yolo, mediapipe = load_trial(trial_folder)
    print(f"  Loaded — Manual: {len(manual)} rows | "
          f"YOLO: {len(yolo)} rows | MediaPipe: {len(mediapipe)} rows")

    if normalise:
        yolo      = normalise_to_reference(yolo,      manual)
        mediapipe = normalise_to_reference(mediapipe, manual)
        print("  Coordinate normalisation applied.")

    # ── Metrics ──────────────────────────────────────────────────────────
    accuracy_df     = compute_accuracy_metrics(manual, yolo, mediapipe)
    reliability_df  = compute_reliability_metrics(manual, yolo, mediapipe)

    acc_path = out_dir / "accuracy_metrics.csv"
    rel_path = out_dir / "reliability_metrics.csv"
    accuracy_df.to_csv(acc_path,    index=False)
    reliability_df.to_csv(rel_path, index=False)
    print(f"  ✓ accuracy_metrics.csv   ({len(accuracy_df)} rows)")
    print(f"  ✓ reliability_metrics.csv ({len(reliability_df)} rows)")

    # ── Plots ─────────────────────────────────────────────────────────────
    plot_error_bar_chart(
        accuracy_df,
        out_dir / "error_bar_chart.png",
        trial_label=trial_label,
    )
    print("  ✓ error_bar_chart.png")

    plot_precision_scatter(
        manual, yolo, mediapipe,
        scatter_joint,
        out_dir / "precision_scatter.png",
        trial_label=trial_label,
    )
    print(f"  ✓ precision_scatter.png  (joint: {scatter_joint})")

    plot_bland_altman(
        manual, yolo, mediapipe,
        out_dir / "bland_altman.png",
        trial_label=trial_label,
    )
    print("  ✓ bland_altman.png")

    plot_summary_dashboard(
        accuracy_df, reliability_df,
        out_dir / "summary_dashboard.png",
        trial_label=trial_label,
    )
    print("  ✓ summary_dashboard.png")

    return accuracy_df, reliability_df


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pose Estimation Accuracy Analysis — RQ1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path to a single trial folder, or to a parent folder (use --batch).",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Output root directory (default: <input>/analysis_output).",
    )
    parser.add_argument(
        "--batch", action="store_true",
        help="Treat --input as a folder of trial sub-folders.",
    )
    parser.add_argument(
        "--scatter-joint", default="R-Ankle",
        help="Joint to use for the precision scatter plot (default: R-Ankle).",
    )
    parser.add_argument(
        "--no-normalise", action="store_true",
        help="Skip coordinate normalisation step.",
    )
    args = parser.parse_args()

    input_path  = Path(args.input).expanduser().resolve()
    output_root = Path(args.output).expanduser().resolve() if args.output \
                  else input_path / "analysis_output"
    output_root.mkdir(parents=True, exist_ok=True)

    normalise = not args.no_normalise

    if args.batch:
        trial_folders = sorted([
            p for p in input_path.iterdir()
            if p.is_dir() and not p.name.startswith(".")
        ])
        if not trial_folders:
            raise SystemExit(f"No sub-folders found in {input_path}")
        print(f"Batch mode: {len(trial_folders)} trial(s) found.")

        acc_list, rel_list = [], []
        for folder in trial_folders:
            try:
                acc, rel = analyse_trial(
                    folder, output_root,
                    scatter_joint=args.scatter_joint,
                    normalise=normalise,
                )
                acc["Trial"] = folder.name
                rel["Trial"] = folder.name
                acc_list.append(acc)
                rel_list.append(rel)
            except (FileNotFoundError, ValueError) as exc:
                print(f"  ⚠ Skipping {folder.name}: {exc}")

        if acc_list:
            acc_agg, rel_agg = aggregate_trials(acc_list, rel_list)
            agg_dir = output_root / "_aggregated"
            agg_dir.mkdir(exist_ok=True)
            acc_agg.to_csv(agg_dir / "accuracy_aggregated.csv", index=False)
            rel_agg.to_csv(agg_dir / "reliability_aggregated.csv", index=False)

            # Aggregated error bar chart
            plot_error_bar_chart(
                acc_agg.rename(columns={
                    "MEE_cm_mean":"MEE_cm","RMSE_cm_mean":"RMSE_cm",
                    "Bias_X_cm_mean":"Bias_X_cm","Bias_Y_cm_mean":"Bias_Y_cm",
                }),
                agg_dir / "error_bar_chart_aggregated.png",
                trial_label=f"Aggregated ({len(acc_list)} trials)",
            )
            print(f"\n✓ Aggregated outputs → {agg_dir}")

    else:
        analyse_trial(
            input_path, output_root,
            scatter_joint=args.scatter_joint,
            normalise=normalise,
        )

    print(f"\n{'='*60}")
    print(f"  All outputs saved to: {output_root}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()