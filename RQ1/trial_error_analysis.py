"""
=============================================================================
Trial-Wise Pose Estimation Error Analysis — Delta_X / Delta_Y
=============================================================================
Computes MSE, RMSE, and MPJPE for YOLO vs Manual and MediaPipe vs Manual,
using Delta_X and Delta_Y (relative displacement from frame-0).

Naming convention expected (all CSVs flat in one folder):
    <PID>_<TID>_Manual.csv
    <PID>_<TID>_YOLO.csv
    <PID>_<TID>_MediaPipe.csv

Usage:
    python trial_error_analysis.py --input /path/to/csvs/

Outputs → <input>/trial_analysis/
    trial_metrics.csv              — MSE / RMSE / MPJPE per trial × model × joint
    trial_summary.csv              — per-trial overall means
    per_joint_rmse_heatmap.png     — RMSE heatmap: trials × joints (separate for each model)
    per_trial_mpjpe_bar.png        — MPJPE bar chart per trial, YOLO vs MediaPipe
    delta_x_rmse_comparison.png    — side-by-side RMSE for Delta_X per joint per trial
    delta_y_rmse_comparison.png    — side-by-side RMSE for Delta_Y per joint per trial
    error_table.png                — clean printable table of all metrics
=============================================================================
"""

from __future__ import annotations

import argparse
import re
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Style
# ─────────────────────────────────────────────────────────────────────────────
PALETTE = {"YOLO": "#FF6B35", "MediaPipe": "#00C9A7"}
JOINT_ORDER = [
    "Head",
    "L-Shoulder", "R-Shoulder",
    "L-Elbow",    "R-Elbow",
    "L-Wrist",    "R-Wrist",
    "L-Knee",     "R-Knee",
    "L-Ankle",    "R-Ankle",
]

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "grid.linestyle":   "--",
    "figure.dpi":       150,
    "savefig.dpi":      150,
    "savefig.bbox":     "tight",
})


# ─────────────────────────────────────────────────────────────────────────────
# Discovery
# ─────────────────────────────────────────────────────────────────────────────

def discover_trials(folder: Path) -> dict[tuple[str, str], dict[str, Path]]:
    pattern = re.compile(
        r"^(?P<pid>.+?)_(?P<tid>[^_]+)_(?P<model>Manual|YOLO|MediaPipe)\.csv$",
        re.IGNORECASE,
    )
    groups: dict[tuple[str, str], dict[str, Path]] = defaultdict(dict)
    for p in sorted(folder.glob("*.csv")):
        m = pattern.match(p.name)
        if not m:
            continue
        pid = m.group("pid").upper()
        tid = m.group("tid").upper()
        raw = m.group("model").lower()
        if   "yolo"      in raw: model = "YOLO"
        elif "mediapipe" in raw: model = "MediaPipe"
        else:                    model = "Manual"
        groups[(pid, tid)][model] = p

    complete = {k: v for k, v in groups.items()
                if {"Manual", "YOLO", "MediaPipe"}.issubset(v)}
    if not complete:
        raise FileNotFoundError(
            "No complete triplets found.\n"
            "Need: <PID>_<TID>_Manual.csv  +  _YOLO.csv  +  _MediaPipe.csv"
        )
    return complete


# ─────────────────────────────────────────────────────────────────────────────
# Load
# ─────────────────────────────────────────────────────────────────────────────

REQUIRED = {"Frame_Num", "Joint_Name", "Delta_X", "Delta_Y"}


def _load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}")
    df["Frame_Num"]  = df["Frame_Num"].astype(int)
    df["Joint_Name"] = df["Joint_Name"].str.strip()
    df["Delta_X"]    = pd.to_numeric(df["Delta_X"], errors="coerce")
    df["Delta_Y"]    = pd.to_numeric(df["Delta_Y"], errors="coerce")
    return df[["Frame_Num", "Joint_Name", "Delta_X", "Delta_Y"]]


# ─────────────────────────────────────────────────────────────────────────────
# Core metric calculations
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(gt: pd.DataFrame,
                    pred: pd.DataFrame,
                    model_label: str,
                    pid: str,
                    tid: str) -> pd.DataFrame:
    """
    Merge GT and predicted on (Frame_Num, Joint_Name).
    Compute per-joint:
        MSE_X, MSE_Y          — mean squared error on Delta_X and Delta_Y (metres²)
        RMSE_X, RMSE_Y        — root MSE (metres)
        RMSE_2D               — √(RMSE_X² + RMSE_Y²) — combined planar RMSE
        MPJPE                 — mean per-joint position error (Euclidean, metres)
                                  = mean( √(ΔX_err² + ΔY_err²) ) per joint

    All errors are in metres (your Delta columns are already in metres).
    Also adds _cm columns (× 100) for readability.
    """
    merged = gt.merge(pred,
                      on=["Frame_Num", "Joint_Name"],
                      suffixes=("_gt", "_pred"))
    merged = merged.dropna(subset=["Delta_X_gt", "Delta_Y_gt",
                                    "Delta_X_pred", "Delta_Y_pred"])

    merged["err_x"]  = merged["Delta_X_pred"] - merged["Delta_X_gt"]
    merged["err_y"]  = merged["Delta_Y_pred"] - merged["Delta_Y_gt"]
    merged["err_2d"] = np.sqrt(merged["err_x"]**2 + merged["err_y"]**2)

    rows = []
    for joint in JOINT_ORDER:
        s = merged[merged["Joint_Name"] == joint]
        if s.empty:
            continue
        n       = len(s)
        mse_x   = float((s["err_x"]**2).mean())
        mse_y   = float((s["err_y"]**2).mean())
        rmse_x  = float(np.sqrt(mse_x))
        rmse_y  = float(np.sqrt(mse_y))
        rmse_2d = float(np.sqrt(rmse_x**2 + rmse_y**2))
        mpjpe   = float(s["err_2d"].mean())

        rows.append(dict(
            Participant = pid,
            Trial       = tid,
            Trial_Label = f"{pid}_{tid}",
            Model       = model_label,
            Joint       = joint,
            N_Frames    = n,
            # metres
            MSE_X_m     = round(mse_x,   8),
            MSE_Y_m     = round(mse_y,   8),
            RMSE_X_m    = round(rmse_x,  6),
            RMSE_Y_m    = round(rmse_y,  6),
            RMSE_2D_m   = round(rmse_2d, 6),
            MPJPE_m     = round(mpjpe,   6),
            # centimetres
            MSE_X_cm2   = round(mse_x   * 10000, 6),   # m² → cm² (×10000)
            MSE_Y_cm2   = round(mse_y   * 10000, 6),
            RMSE_X_cm   = round(rmse_x  * 100,   4),
            RMSE_Y_cm   = round(rmse_y  * 100,   4),
            RMSE_2D_cm  = round(rmse_2d * 100,   4),
            MPJPE_cm    = round(mpjpe   * 100,   4),
        ))
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_mpjpe_bar(df: pd.DataFrame, out: Path) -> None:
    """
    Grouped bar chart: MPJPE (cm) per trial, YOLO vs MediaPipe.
    Each group = one trial, two bars side by side.
    """
    trial_model = (df.groupby(["Trial_Label", "Model"])["MPJPE_cm"]
                   .mean().reset_index())
    trials  = sorted(trial_model["Trial_Label"].unique())
    x       = np.arange(len(trials))
    w       = 0.35

    fig, ax = plt.subplots(figsize=(max(10, len(trials) * 1.4), 6))
    for i, model in enumerate(["YOLO", "MediaPipe"]):
        sub  = trial_model[trial_model.Model == model].set_index("Trial_Label")
        vals = [sub.loc[t, "MPJPE_cm"] if t in sub.index else np.nan for t in trials]
        bars = ax.bar(x + i * w - w / 2, vals, w * 0.9,
                      color=PALETTE[model], alpha=0.88, label=model, zorder=3)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.002,
                        f"{v:.3f}", ha="center", va="bottom",
                        fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(trials, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("MPJPE (cm)", fontsize=11)
    ax.set_title("Mean Per-Joint Position Error (MPJPE) — Trial-wise Comparison",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"  ✓  {out.name}")


def plot_rmse_heatmap(df: pd.DataFrame, metric: str,
                      label: str, out: Path) -> None:
    """
    Two heatmaps side-by-side (YOLO | MediaPipe).
    Rows = trials, columns = joints, cells = chosen metric.
    """
    n_trials = df["Trial_Label"].nunique()
    fig, axes = plt.subplots(1, 2,
                              figsize=(22, max(5, n_trials * 0.75 + 2)))
    for ax, model in zip(axes, ["YOLO", "MediaPipe"]):
        sub  = df[df["Model"] == model]
        heat = sub.pivot_table(index="Trial_Label", columns="Joint",
                               values=metric, aggfunc="mean")
        heat = heat.reindex(columns=[j for j in JOINT_ORDER if j in heat.columns])
        heat = heat.reindex(sorted(heat.index))

        vmax = heat.max().max()
        sns.heatmap(heat, ax=ax, annot=True, fmt=".3f",
                    cmap="YlOrRd", vmin=0, vmax=vmax,
                    linewidths=0.4,
                    cbar_kws={"label": f"{label} (cm)"},
                    annot_kws={"size": 8.5})
        ax.set_title(f"{model} — {label} (cm)",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("Joint", fontsize=9)
        ax.set_ylabel("Trial", fontsize=9)
        ax.tick_params(axis="x", labelsize=8, rotation=38)
        ax.tick_params(axis="y", labelsize=9)

    fig.suptitle(f"{label} per Trial × Joint — YOLO vs MediaPipe",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"  ✓  {out.name}")


def plot_delta_rmse_comparison(df: pd.DataFrame, axis: str, out: Path) -> None:
    """
    For each trial: grouped bars of RMSE_{axis}_cm per joint, YOLO vs MediaPipe.
    One subplot per trial, all on one figure.
    """
    metric  = f"RMSE_{axis}_cm"
    trials  = sorted(df["Trial_Label"].unique())
    ncols   = min(3, len(trials))
    nrows   = int(np.ceil(len(trials) / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols * 7, nrows * 4.5),
                              squeeze=False)
    axes_flat = axes.flatten()

    for idx, trial in enumerate(trials):
        ax      = axes_flat[idx]
        sub     = df[df["Trial_Label"] == trial]
        joints  = [j for j in JOINT_ORDER if j in sub["Joint"].values]
        x       = np.arange(len(joints))
        w       = 0.35

        for i, model in enumerate(["YOLO", "MediaPipe"]):
            ms   = sub[sub["Model"] == model].set_index("Joint")
            vals = [ms.loc[j, metric] if j in ms.index else np.nan for j in joints]
            ax.bar(x + i * w - w / 2, vals, w * 0.9,
                   color=PALETTE[model], alpha=0.88, label=model, zorder=3)

        ax.set_xticks(x)
        ax.set_xticklabels(joints, rotation=38, ha="right", fontsize=7.5)
        ax.set_ylabel(f"RMSE Δ{axis} (cm)", fontsize=9)
        ax.set_title(trial, fontsize=10, fontweight="bold")
        if idx == 0:
            ax.legend(fontsize=8)

    # Hide unused subplots
    for idx in range(len(trials), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(f"RMSE on Delta_{axis} — Per Trial × Joint",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"  ✓  {out.name}")


def plot_error_table(summary: pd.DataFrame, out: Path) -> None:
    """
    Clean printable table: Trial | Model | MPJPE | RMSE_X | RMSE_Y | RMSE_2D
    """
    cols_show = ["Trial_Label", "Model",
                 "MPJPE_cm", "RMSE_X_cm", "RMSE_Y_cm", "RMSE_2D_cm",
                 "MSE_X_cm2", "MSE_Y_cm2"]
    tbl = (summary.groupby(["Trial_Label", "Model"])[
                    ["MPJPE_cm","RMSE_X_cm","RMSE_Y_cm",
                     "RMSE_2D_cm","MSE_X_cm2","MSE_Y_cm2"]]
           .mean().round(4).reset_index())
    tbl = tbl.sort_values(["Trial_Label", "Model"])

    col_labels = ["Trial", "Model",
                  "MPJPE\n(cm)", "RMSE-ΔX\n(cm)", "RMSE-ΔY\n(cm)",
                  "RMSE-2D\n(cm)", "MSE-ΔX\n(cm²)", "MSE-ΔY\n(cm²)"]

    n_rows  = len(tbl)
    fig_h   = max(4, n_rows * 0.45 + 1.8)
    fig, ax = plt.subplots(figsize=(18, fig_h))
    ax.axis("off")

    table = ax.table(
        cellText   = tbl.values,
        colLabels  = col_labels,
        loc        = "center",
        cellLoc    = "center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.8)

    # Header style
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#1a1a2e")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Row colouring by model
    for i in range(1, n_rows + 1):
        model = tbl.iloc[i - 1]["Model"]
        bg    = PALETTE.get(model, "#f5f5f5") + "28"
        for j in range(len(col_labels)):
            table[i, j].set_facecolor(bg)

    # Alternate trial rows slightly
    prev_trial = None
    shade = False
    for i in range(1, n_rows + 1):
        trial = tbl.iloc[i - 1]["Trial_Label"]
        if trial != prev_trial:
            shade = not shade
            prev_trial = trial
        if shade:
            for j in range(len(col_labels)):
                existing = table[i, j].get_facecolor()
                # Darken slightly
                r, g, b, a = existing
                table[i, j].set_facecolor((r * 0.92, g * 0.92, b * 0.92, a))

    fig.suptitle("Trial-wise Error Summary — Delta_X / Delta_Y  (YOLO vs MediaPipe vs Manual GT)",
                 fontsize=12, fontweight="bold", y=0.98)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {out.name}")


def plot_overall_joint_comparison(df: pd.DataFrame, out: Path) -> None:
    """
    Line chart: MPJPE per joint, one line per (trial × model).
    Shows consistency of errors across joints across trials.
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    trials = sorted(df["Trial_Label"].unique())

    cmap   = plt.get_cmap("tab10")
    colors = {t: cmap(i / max(len(trials) - 1, 1)) for i, t in enumerate(trials)}

    for ax, model in zip(axes, ["YOLO", "MediaPipe"]):
        for trial in trials:
            sub    = df[(df["Model"] == model) & (df["Trial_Label"] == trial)]
            joints = [j for j in JOINT_ORDER if j in sub["Joint"].values]
            vals   = [sub[sub["Joint"] == j]["MPJPE_cm"].values[0]
                      if not sub[sub["Joint"] == j].empty else np.nan
                      for j in joints]
            ax.plot(range(len(joints)), vals,
                    marker="o", linewidth=1.6, markersize=5,
                    label=trial, color=colors[trial], alpha=0.85)

        ax.set_xticks(range(len(JOINT_ORDER)))
        ax.set_xticklabels(JOINT_ORDER, rotation=38, ha="right", fontsize=9)
        ax.set_ylabel("MPJPE (cm)", fontsize=10)
        ax.set_title(f"{model} — MPJPE per Joint, all Trials",
                     fontsize=11, fontweight="bold")
        ax.legend(fontsize=8, title="Trial", ncol=2)

    fig.suptitle("Joint-wise MPJPE — All Trials Overlaid",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"  ✓  {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run(input_dir: Path, output_dir: Path) -> None:
    print(f"\n{'═'*60}")
    print(f"  Trial-Wise Delta Error Analysis")
    print(f"  Input  : {input_dir}")
    print(f"  Output : {output_dir}")
    print(f"{'═'*60}\n")

    trials = discover_trials(input_dir)
    print(f"  Found {len(trials)} trial(s):\n")
    for pid, tid in sorted(trials):
        print(f"    {pid}  {tid}")
    print()

    all_metrics: list[pd.DataFrame] = []

    for (pid, tid) in sorted(trials):
        paths  = trials[(pid, tid)]
        label  = f"{pid}_{tid}"
        print(f"  Computing metrics for {label} …")

        manual    = _load(paths["Manual"])
        yolo      = _load(paths["YOLO"])
        mediapipe = _load(paths["MediaPipe"])

        metrics_y  = compute_metrics(manual, yolo,      "YOLO",      pid, tid)
        metrics_mp = compute_metrics(manual, mediapipe, "MediaPipe", pid, tid)

        all_metrics.extend([metrics_y, metrics_mp])
        print(f"    ✓  {label}: {len(metrics_y)} joint rows × 2 models")

    df = pd.concat(all_metrics, ignore_index=True)

    # ── CSVs ─────────────────────────────────────────────────────────────
    df.to_csv(output_dir / "trial_metrics.csv", index=False)
    print(f"\n  ✓  trial_metrics.csv  ({len(df)} rows)")

    summary = (df.groupby(["Trial_Label", "Model"])
               [["MPJPE_cm","RMSE_X_cm","RMSE_Y_cm","RMSE_2D_cm",
                 "MSE_X_cm2","MSE_Y_cm2"]]
               .mean().round(4).reset_index())
    summary.to_csv(output_dir / "trial_summary.csv", index=False)
    print(f"  ✓  trial_summary.csv  ({len(summary)} rows)")

    # ── Plots ─────────────────────────────────────────────────────────────
    print(f"\n  Generating plots …\n")

    plot_mpjpe_bar(df,       output_dir / "per_trial_mpjpe_bar.png")
    plot_rmse_heatmap(df, "RMSE_X_cm",  "RMSE ΔX", output_dir / "rmse_delta_x_heatmap.png")
    plot_rmse_heatmap(df, "RMSE_Y_cm",  "RMSE ΔY", output_dir / "rmse_delta_y_heatmap.png")
    plot_rmse_heatmap(df, "RMSE_2D_cm", "RMSE 2D", output_dir / "rmse_2d_heatmap.png")
    plot_rmse_heatmap(df, "MPJPE_cm",   "MPJPE",   output_dir / "mpjpe_heatmap.png")
    plot_delta_rmse_comparison(df, "X", output_dir / "delta_x_rmse_comparison.png")
    plot_delta_rmse_comparison(df, "Y", output_dir / "delta_y_rmse_comparison.png")
    plot_overall_joint_comparison(df,   output_dir / "joint_mpjpe_all_trials.png")
    plot_error_table(df,                output_dir / "error_table.png")

    # ── Console summary ───────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  OVERALL SUMMARY (mean across all trials & joints)")
    print(f"{'─'*60}")
    overall = (df.groupby("Model")
               [["MPJPE_cm","RMSE_X_cm","RMSE_Y_cm","RMSE_2D_cm"]]
               .mean().round(4))
    print(overall.to_string())
    print(f"{'─'*60}")

    print(f"\n{'═'*60}")
    print(f"  DONE")
    print(f"  Output : {output_dir}")
    print(f"  Files  :")
    for f in sorted(output_dir.iterdir()):
        print(f"    {f.name}")
    print(f"{'═'*60}\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Trial-wise MSE / RMSE / MPJPE on Delta_X and Delta_Y",
    )
    parser.add_argument("--input",  "-i", required=True,
                        help="Folder with all P##_T##_Manual/YOLO/MediaPipe.csv files")
    parser.add_argument("--output", "-o", default=None,
                        help="Output folder (default: <input>/trial_analysis/)")
    args = parser.parse_args()

    inp = Path(args.input).expanduser().resolve()
    out = (Path(args.output).expanduser().resolve()
           if args.output else inp / "trial_analysis")
    out.mkdir(parents=True, exist_ok=True)

    run(inp, out)