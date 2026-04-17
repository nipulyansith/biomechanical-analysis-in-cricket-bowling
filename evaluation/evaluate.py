"""
Evaluation Module
=================
Compares ground_truth.csv (manual annotations) vs master.xlsx (model outputs).
Computes MAE and RMSE per parameter, prints a table, and saves a comparison plot.

Usage:
    python evaluate.py [--gt ground_truth.csv] [--model master.xlsx]
"""

import argparse
import ast
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

warnings.filterwarnings("ignore")

# ── numeric columns shared between the two datasets ──────────────────────────
NUMERIC_PARAMS = [
    ("stride_duration_s",           "s"),
    ("stride_length_m",             "m"),
    ("step_duration_mean_s",        "s"),
    ("step_duration_std_s",         "s"),
    ("step_duration_cv",            ""),
    ("final5_total_duration_s",     "s"),
    ("elbow_angle_arm_back_deg",    "deg"),
    ("elbow_angle_release_deg",     "deg"),
    ("elbow_extension_deg",         "deg"),
    ("knee_angle_ffc_deg",          "deg"),
    ("knee_angle_release_deg",      "deg"),
    ("head_dx_ffc_cm",              "cm"),
    ("head_dy_ffc_cm",              "cm"),
    ("head_d_ffc_cm",               "cm"),
    ("head_dx_bfc_cm",              "cm"),
    ("head_dy_bfc_cm",              "cm"),
    ("head_d_bfc_cm",               "cm"),
    ("wrist_speed_at_release_m_s",  "m/s"),
]

FRAME_PARAMS = [
    ("bfc_frame",       "frames"),
    ("ffc_frame",       "frames"),
    ("arm_back_frame",  "frames"),
    ("release_frame.1", "frames"),
]

ALL_PARAMS = NUMERIC_PARAMS + FRAME_PARAMS


def safe_float(v):
    """Return float or NaN."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    try:
        return float(v)
    except Exception:
        return np.nan


def load_model(path):
    df = pd.read_excel(path)
    return df


def load_gt(path):
    df = pd.read_csv(path)
    return df


def align(model_df, gt_df):
    """Inner-join on trial_id."""
    merged = pd.merge(gt_df, model_df, on="trial_id", suffixes=("_gt", "_model"))
    return merged


def compute_metrics(merged, params):
    rows = []
    for col, unit in params:
        gt_col    = col + "_gt"    if (col + "_gt")    in merged.columns else col
        model_col = col + "_model" if (col + "_model") in merged.columns else col

        if gt_col not in merged.columns or model_col not in merged.columns:
            continue

        gt_vals    = merged[gt_col].apply(safe_float)
        model_vals = merged[model_col].apply(safe_float)

        mask = gt_vals.notna() & model_vals.notna()
        if mask.sum() == 0:
            continue

        errors = (gt_vals[mask] - model_vals[mask]).values
        mae    = float(np.mean(np.abs(errors)))
        rmse   = float(np.sqrt(np.mean(errors ** 2)))
        n      = int(mask.sum())

        rows.append({
            "Parameter": col,
            "N":         n,
            "MAE":       round(mae, 4),
            "RMSE":      round(rmse, 4),
            "Unit":      unit,
        })

    return pd.DataFrame(rows)


def print_table(metrics_df):
    print("\n" + "="*72)
    print(f"{'Parameter':<35} {'N':>4} {'MAE':>10} {'RMSE':>10}  Unit")
    print("-"*72)
    for _, row in metrics_df.iterrows():
        print(f"{row['Parameter']:<35} {row['N']:>4} {row['MAE']:>10.4f} {row['RMSE']:>10.4f}  {row['Unit']}")
    print("="*72)


def plot_comparison(merged, out_path="evaluation_plot.png"):
    """Plot manual vs model for elbow_angle_release_deg (+ a few others)."""
    plot_cols = [
        ("elbow_angle_release_deg", "Elbow Angle at Release (deg)"),
        ("elbow_angle_arm_back_deg","Elbow Angle at Arm Back (deg)"),
        ("knee_angle_ffc_deg",      "Knee Angle at FFC (deg)"),
        ("stride_length_m",         "Stride Length (m)"),
        ("wrist_speed_at_release_m_s", "Wrist Speed at Release (m/s)"),
    ]

    available = [(c, lbl) for c, lbl in plot_cols
                 if (c+"_gt") in merged.columns and (c+"_model") in merged.columns]

    n = len(available)
    if n == 0:
        print("  No overlapping columns to plot.")
        return

    fig, axes = plt.subplots(1, n, figsize=(5*n, 5))
    if n == 1:
        axes = [axes]

    fig.suptitle("Manual Ground Truth vs Model Outputs", fontsize=14, fontweight="bold", y=1.01)

    for ax, (col, lbl) in zip(axes, available):
        gt_v    = merged[col+"_gt"].apply(safe_float)
        model_v = merged[col+"_model"].apply(safe_float)
        mask    = gt_v.notna() & model_v.notna()

        if mask.sum() == 0:
            ax.set_title(lbl)
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue

        x = gt_v[mask].values
        y = model_v[mask].values

        # scatter
        ax.scatter(x, y, color="#2563EB", s=80, zorder=3, label="Trials")

        # identity line
        lo = min(x.min(), y.min()) * 0.95
        hi = max(x.max(), y.max()) * 1.05
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="Perfect agreement")

        # annotate trial ids
        for i, (xi, yi) in enumerate(zip(x, y)):
            tid = merged["trial_id"].iloc[list(mask[mask].index).index(mask[mask].index[i])] \
                  if "trial_id" in merged.columns else ""
            ax.annotate(str(tid), (xi, yi), fontsize=7, textcoords="offset points", xytext=(5, 3))

        mae  = float(np.mean(np.abs(x - y)))
        rmse = float(np.sqrt(np.mean((x - y)**2)))

        ax.set_xlabel("Manual (Ground Truth)", fontsize=10)
        ax.set_ylabel("Model Output",          fontsize=10)
        ax.set_title(lbl, fontsize=10, fontweight="bold")
        ax.legend(fontsize=8)
        ax.text(0.03, 0.97, f"MAE={mae:.3f}\nRMSE={rmse:.3f}",
                transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
        ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Plot saved → {out_path}")
    plt.close()


def save_metrics(metrics_df, path="evaluation_metrics.csv"):
    metrics_df.to_csv(path, index=False)
    print(f"  Metrics saved → {path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate manual vs model cricket biomechanics")
    parser.add_argument("--gt",    default="ground_truth.csv",  help="Ground truth CSV")
    parser.add_argument("--model", default="master.xlsx",        help="Model outputs XLSX")
    parser.add_argument("--plot",  default="evaluation_plot.png",help="Output plot path")
    args = parser.parse_args()

    print(f"\nLoading model  : {args.model}")
    print(f"Loading GT     : {args.gt}")

    model_df = load_model(args.model)
    gt_df    = load_gt(args.gt)

    print(f"  Model rows : {len(model_df)}")
    print(f"  GT rows    : {len(gt_df)}")

    merged = align(model_df, gt_df)
    print(f"  Matched    : {len(merged)} trial(s)")

    if len(merged) == 0:
        print("  No matching trial_ids found. Nothing to evaluate.")
        return

    metrics = compute_metrics(merged, ALL_PARAMS)
    print_table(metrics)
    save_metrics(metrics)
    plot_comparison(merged, args.plot)
    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
