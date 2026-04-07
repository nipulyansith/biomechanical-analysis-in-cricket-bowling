"""
compare_models.py
-----------------
Loads  output/nmodel.csv, output/lmodel.csv, output/groundtruth.csv
and produces a multi-page PDF + individual PNGs showing exactly how
each model deviates from ground truth.

Graphs produced:
  1. Per-keypoint Mean Absolute Error  (N vs L — bar chart)
  2. Per-frame overall MAE over time   (line chart)
  3. Scatter: predicted vs GT  (X and Y axes, both models, all keypoints)
  4. Error heatmap  (frame × keypoint)  for N-model
  5. Error heatmap  (frame × keypoint)  for L-model
  6. Per-keypoint X-error and Y-error box plots
  7. Detection rate per keypoint (% of frames where model had valid coords)
  8. Error vector quiver plot (shows direction & magnitude of error)

Run:
    python compare_models.py

Output:
    output/comparison_report.pdf
    output/plots/  (individual PNGs)
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
NMODEL_CSV = "output/nmodel.csv"
LMODEL_CSV = "output/lmodel.csv"
GT_CSV     = "output/groundtruth.csv"
OUT_PDF    = "output/comparison_report.pdf"
OUT_DIR    = "output/plots"

KEYPOINTS_8 = [
    "left_shoulder",  "right_shoulder",
    "left_elbow",     "right_elbow",
    "left_wrist",     "right_wrist",
    "left_ankle",     "right_ankle",
]

# Short display labels
KP_SHORT = {
    "left_shoulder":  "L.Shoulder", "right_shoulder": "R.Shoulder",
    "left_elbow":     "L.Elbow",    "right_elbow":    "R.Elbow",
    "left_wrist":     "L.Wrist",    "right_wrist":    "R.Wrist",
    "left_ankle":     "L.Ankle",    "right_ankle":    "R.Ankle",
}

# ── Visual style ─────────────────────────────────────────────────────────────
DARK_BG  = "#0d0d14"
SURFACE  = "#16161f"
BORDER   = "#2a2a3d"
N_COLOR  = "#ff6b6b"   # red — N-model (smaller / less accurate)
L_COLOR  = "#54a0ff"   # blue — L-model (larger / more accurate)
GT_COLOR = "#3ddc84"   # green — ground truth
ACCENT   = "#feca57"
TEXT     = "#e8e8f0"
MUTED    = "#6b6b85"

plt.rcParams.update({
    "figure.facecolor":  DARK_BG,
    "axes.facecolor":    SURFACE,
    "axes.edgecolor":    BORDER,
    "axes.labelcolor":   TEXT,
    "xtick.color":       MUTED,
    "ytick.color":       MUTED,
    "text.color":        TEXT,
    "grid.color":        BORDER,
    "grid.linewidth":    0.5,
    "font.family":       "monospace",
    "font.size":         10,
    "axes.titlesize":    12,
    "axes.titleweight":  "bold",
    "legend.framealpha": 0.15,
    "legend.edgecolor":  BORDER,
    "figure.dpi":        120,
})

# Custom heatmap colormap: dark → orange → red
HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "heat", ["#0d0d14", "#533483", "#ff6b6b", "#feca57"], N=256
)
# ─────────────────────────────────────────────────────────────────────────────


def load_data():
    """Load all three CSVs and align on common frames."""
    gt = pd.read_csv(GT_CSV)
    nm = pd.read_csv(NMODEL_CSV)
    lm = pd.read_csv(LMODEL_CSV)

    # Merge on frame — only keep frames present in ground truth
    merged = gt.copy()
    merged = merged.merge(nm, on="frame", suffixes=("", "_n"), how="left")
    merged = merged.merge(lm, on="frame", suffixes=("", "_l"), how="left")
    merged = merged.sort_values("frame").reset_index(drop=True)
    return merged


def compute_errors(merged):
    """
    Returns two dicts, one per model:
      errors["n"][kp_name] = {"ex": array, "ey": array, "dist": array}
    Only frames where BOTH GT and model have valid coords are included.
    """
    results = {"n": {}, "l": {}}
    for kp in KEYPOINTS_8:
        gt_x  = merged[f"{kp}_x"].values
        gt_y  = merged[f"{kp}_y"].values
        for tag in ("n", "l"):
            suffix = "" if tag == "n" else "_l"
            # N-model columns were merged without suffix (gt was base); L got _l
            if tag == "n":
                px = merged.get(f"{kp}_x_n", merged.get(f"{kp}_x_n"))
                py = merged.get(f"{kp}_y_n", merged.get(f"{kp}_y_n"))
                # nmodel columns: after merging gt (base) + nm (suffix='_n' for conflicts)
                # if column name didn't conflict it keeps original name
                col_x = f"{kp}_x_n" if f"{kp}_x_n" in merged.columns else f"{kp}_x"
                col_y = f"{kp}_y_n" if f"{kp}_y_n" in merged.columns else f"{kp}_y"
                px = merged[col_x].values
                py = merged[col_y].values
            else:
                col_x = f"{kp}_x_l"
                col_y = f"{kp}_y_l"
                if col_x not in merged.columns:
                    col_x = f"{kp}_x"
                    col_y = f"{kp}_y"
                px = merged[col_x].values
                py = merged[col_y].values

            valid = (~np.isnan(gt_x)) & (~np.isnan(gt_y)) & (~np.isnan(px)) & (~np.isnan(py))
            ex    = (px - gt_x)[valid]
            ey    = (py - gt_y)[valid]
            dist  = np.sqrt(ex**2 + ey**2)
            results[tag][kp] = {
                "ex": ex, "ey": ey, "dist": dist,
                "valid_mask": valid,
                "frames": merged["frame"].values[valid],
                "gt_x": gt_x[valid], "gt_y": gt_y[valid],
                "pred_x": px[valid], "pred_y": py[valid],
            }
    return results


def detection_rates(merged):
    """% of ground-truth frames where each model produced a valid detection."""
    rates = {"n": {}, "l": {}}
    for kp in KEYPOINTS_8:
        gt_valid = merged[f"{kp}_x"].notna()
        for tag, suffix in [("n", "_n"), ("l", "_l")]:
            col = f"{kp}_x{suffix}"
            if col not in merged.columns:
                col = f"{kp}_x"
            model_valid = merged[col].notna() & gt_valid
            rates[tag][kp] = 100.0 * model_valid.sum() / max(gt_valid.sum(), 1)
    return rates


def fig_save(fig, pdf, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(os.path.join(OUT_DIR, f"{name}.png"), bbox_inches="tight",
                facecolor=DARK_BG, dpi=150)
    pdf.savefig(fig, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  ✓ {name}.png")


# ═══════════════════════════════════════════════════════════════════════════
# PLOT 1 — Per-keypoint Mean Absolute Error (bar chart)
# ═══════════════════════════════════════════════════════════════════════════
def plot_per_kp_mae(errors, pdf):
    kp_labels = [KP_SHORT[k] for k in KEYPOINTS_8]
    n_mae = [errors["n"][k]["dist"].mean() if len(errors["n"][k]["dist"]) else np.nan for k in KEYPOINTS_8]
    l_mae = [errors["l"][k]["dist"].mean() if len(errors["l"][k]["dist"]) else np.nan for k in KEYPOINTS_8]

    fig, ax = plt.subplots(figsize=(13, 5))
    fig.patch.set_facecolor(DARK_BG)

    x    = np.arange(len(KEYPOINTS_8))
    w    = 0.35
    bars_n = ax.bar(x - w/2, n_mae, w, color=N_COLOR, alpha=0.85, label="N-model", zorder=3)
    bars_l = ax.bar(x + w/2, l_mae, w, color=L_COLOR, alpha=0.85, label="L-model", zorder=3)

    # Value labels
    for bar in bars_n:
        if not np.isnan(bar.get_height()):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{bar.get_height():.1f}", ha="center", va="bottom",
                    fontsize=8, color=N_COLOR)
    for bar in bars_l:
        if not np.isnan(bar.get_height()):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{bar.get_height():.1f}", ha="center", va="bottom",
                    fontsize=8, color=L_COLOR)

    ax.set_xticks(x)
    ax.set_xticklabels(kp_labels, rotation=30, ha="right")
    ax.set_ylabel("Mean Euclidean Error (pixels)")
    ax.set_title("Per-Keypoint Mean Error — N-model vs L-model vs Ground Truth")
    ax.legend(loc="upper right")
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig_save(fig, pdf, "01_per_kp_mae")


# ═══════════════════════════════════════════════════════════════════════════
# PLOT 2 — Per-frame overall MAE over time
# ═══════════════════════════════════════════════════════════════════════════
def plot_per_frame_mae(merged, errors, pdf):
    frames_all = merged["frame"].values

    def frame_mae(tag):
        per_frame = {}
        for kp in KEYPOINTS_8:
            e = errors[tag][kp]
            for f, d in zip(e["frames"], e["dist"]):
                per_frame.setdefault(f, []).append(d)
        return {f: np.mean(v) for f, v in per_frame.items()}

    n_fm = frame_mae("n")
    l_fm = frame_mae("l")
    common_frames = sorted(set(n_fm) & set(l_fm))
    fn = [f for f in common_frames]
    n_vals = [n_fm[f] for f in fn]
    l_vals = [l_fm[f] for f in fn]

    fig, ax = plt.subplots(figsize=(14, 4))
    fig.patch.set_facecolor(DARK_BG)

    ax.plot(fn, n_vals, color=N_COLOR, lw=1.5, alpha=0.9, label="N-model", zorder=3)
    ax.plot(fn, l_vals, color=L_COLOR, lw=1.5, alpha=0.9, label="L-model", zorder=3)
    ax.fill_between(fn, n_vals, l_vals,
                    where=[n > l for n, l in zip(n_vals, l_vals)],
                    alpha=0.12, color=N_COLOR, label="N worse")
    ax.fill_between(fn, n_vals, l_vals,
                    where=[l > n for n, l in zip(n_vals, l_vals)],
                    alpha=0.12, color=L_COLOR, label="L worse")

    ax.set_xlabel("Frame number")
    ax.set_ylabel("Mean Error across all keypoints (px)")
    ax.set_title("Per-Frame Mean Euclidean Error Over Time")
    ax.legend(loc="upper right")
    ax.grid(zorder=0)
    fig.tight_layout()
    fig_save(fig, pdf, "02_per_frame_mae")


# ═══════════════════════════════════════════════════════════════════════════
# PLOT 3 — Predicted vs Ground Truth scatter (X and Y)
# ═══════════════════════════════════════════════════════════════════════════
def plot_scatter(errors, pdf):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor(DARK_BG)

    for ax, axis, axis_key in zip(axes, ["X", "Y"], ["x", "y"]):
        for tag, color, label in [("n", N_COLOR, "N-model"), ("l", L_COLOR, "L-model")]:
            gt_all   = np.concatenate([errors[tag][k][f"gt_{axis_key}"]   for k in KEYPOINTS_8])
            pred_all = np.concatenate([errors[tag][k][f"pred_{axis_key}"] for k in KEYPOINTS_8])
            ax.scatter(gt_all, pred_all, color=color, s=5, alpha=0.35,
                       label=label, zorder=3)

        lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
        hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([lo, hi], [lo, hi], color=GT_COLOR, lw=1.2, ls="--",
                label="Perfect", zorder=4)
        ax.set_xlabel(f"Ground Truth {axis} (px)")
        ax.set_ylabel(f"Predicted {axis} (px)")
        ax.set_title(f"Predicted vs GT — {axis} coordinate")
        ax.legend(markerscale=3, loc="upper left")
        ax.grid(zorder=0)

    fig.suptitle("Scatter: All Keypoints — Both Models vs Ground Truth", y=1.01)
    fig.tight_layout()
    fig_save(fig, pdf, "03_scatter_pred_vs_gt")


# ═══════════════════════════════════════════════════════════════════════════
# PLOT 4 & 5 — Error heatmap per model
# ═══════════════════════════════════════════════════════════════════════════
def plot_heatmap(errors, merged, pdf, tag, label, fname):
    all_frames = sorted({f for kp in KEYPOINTS_8 for f in errors[tag][kp]["frames"]})
    matrix     = np.full((len(KEYPOINTS_8), len(all_frames)), np.nan)

    frame_idx_map = {f: i for i, f in enumerate(all_frames)}
    for ki, kp in enumerate(KEYPOINTS_8):
        for f, d in zip(errors[tag][kp]["frames"], errors[tag][kp]["dist"]):
            matrix[ki, frame_idx_map[f]] = d

    fig, ax = plt.subplots(figsize=(max(10, len(all_frames)*0.25 + 2), 5))
    fig.patch.set_facecolor(DARK_BG)

    im = ax.imshow(matrix, aspect="auto", cmap=HEATMAP_CMAP,
                   interpolation="nearest", vmin=0)
    plt.colorbar(im, ax=ax, label="Euclidean error (px)", pad=0.01)

    ax.set_yticks(range(len(KEYPOINTS_8)))
    ax.set_yticklabels([KP_SHORT[k] for k in KEYPOINTS_8])

    # X-tick: show every Nth frame label
    step = max(1, len(all_frames) // 12)
    ax.set_xticks(range(0, len(all_frames), step))
    ax.set_xticklabels([str(all_frames[i]) for i in range(0, len(all_frames), step)],
                       rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Frame number")
    ax.set_title(f"Error Heatmap — {label}   (NaN = no detection)")
    fig.tight_layout()
    fig_save(fig, pdf, fname)


# ═══════════════════════════════════════════════════════════════════════════
# PLOT 6 — Box plots of X-error and Y-error per keypoint
# ═══════════════════════════════════════════════════════════════════════════
def plot_boxplots(errors, pdf):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor(DARK_BG)

    kp_labels = [KP_SHORT[k] for k in KEYPOINTS_8]

    for ax, err_key, axis_label in zip(axes, ["ex", "ey"], ["X-error", "Y-error"]):
        n_data = [errors["n"][k][err_key] for k in KEYPOINTS_8]
        l_data = [errors["l"][k][err_key] for k in KEYPOINTS_8]

        positions = np.arange(len(KEYPOINTS_8))
        w = 0.28

        bp_n = ax.boxplot(n_data, positions=positions - w/2, widths=w,
                          patch_artist=True, manage_ticks=False,
                          medianprops=dict(color="white", linewidth=1.5),
                          whiskerprops=dict(color=N_COLOR, linewidth=0.8),
                          capprops=dict(color=N_COLOR, linewidth=0.8),
                          flierprops=dict(marker=".", color=N_COLOR, alpha=0.3, markersize=3))
        bp_l = ax.boxplot(l_data, positions=positions + w/2, widths=w,
                          patch_artist=True, manage_ticks=False,
                          medianprops=dict(color="white", linewidth=1.5),
                          whiskerprops=dict(color=L_COLOR, linewidth=0.8),
                          capprops=dict(color=L_COLOR, linewidth=0.8),
                          flierprops=dict(marker=".", color=L_COLOR, alpha=0.3, markersize=3))
        for patch in bp_n["boxes"]:
            patch.set(facecolor=N_COLOR, alpha=0.55)
        for patch in bp_l["boxes"]:
            patch.set(facecolor=L_COLOR, alpha=0.55)

        ax.axhline(0, color=GT_COLOR, lw=1, ls="--", alpha=0.6, label="Zero error")
        ax.set_xticks(positions)
        ax.set_xticklabels(kp_labels, rotation=30, ha="right")
        ax.set_ylabel(f"{axis_label} (px)   [positive = model predicts right/down of GT]")
        ax.set_title(f"{axis_label} Distribution per Keypoint")
        ax.grid(axis="y", zorder=0)
        n_patch = mpatches.Patch(color=N_COLOR, alpha=0.7, label="N-model")
        l_patch = mpatches.Patch(color=L_COLOR, alpha=0.7, label="L-model")
        ax.legend(handles=[n_patch, l_patch, mpatches.Patch(color=GT_COLOR, label="Zero error")],
                  loc="upper right")

    fig.suptitle("Error Distribution per Keypoint (signed X and Y)")
    fig.tight_layout()
    fig_save(fig, pdf, "06_boxplots_xy_error")


# ═══════════════════════════════════════════════════════════════════════════
# PLOT 7 — Detection rate bar chart
# ═══════════════════════════════════════════════════════════════════════════
def plot_detection_rate(rates, pdf):
    kp_labels = [KP_SHORT[k] for k in KEYPOINTS_8]
    n_rates   = [rates["n"][k] for k in KEYPOINTS_8]
    l_rates   = [rates["l"][k] for k in KEYPOINTS_8]

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor(DARK_BG)

    x = np.arange(len(KEYPOINTS_8))
    w = 0.35
    ax.bar(x - w/2, n_rates, w, color=N_COLOR, alpha=0.8, label="N-model", zorder=3)
    ax.bar(x + w/2, l_rates, w, color=L_COLOR, alpha=0.8, label="L-model", zorder=3)
    ax.axhline(100, color=GT_COLOR, lw=1, ls="--", alpha=0.5, label="100% detection")
    ax.set_xticks(x)
    ax.set_xticklabels(kp_labels, rotation=30, ha="right")
    ax.set_ylim(0, 110)
    ax.set_ylabel("Detection rate (%)")
    ax.set_title("Keypoint Detection Rate — N-model vs L-model\n(out of frames where ground truth has valid annotation)")
    ax.legend()
    ax.grid(axis="y", zorder=0)
    fig.tight_layout()
    fig_save(fig, pdf, "07_detection_rate")


# ═══════════════════════════════════════════════════════════════════════════
# PLOT 8 — Quiver: error vector map  (one subplot per keypoint)
# ═══════════════════════════════════════════════════════════════════════════
def plot_quiver(errors, pdf):
    ncols = 4
    nrows = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 9))
    fig.patch.set_facecolor(DARK_BG)
    axes = axes.flatten()

    for ki, kp in enumerate(KEYPOINTS_8):
        ax = axes[ki]
        for tag, color, label in [("n", N_COLOR, "N"), ("l", L_COLOR, "L")]:
            e = errors[tag][kp]
            if len(e["gt_x"]) == 0:
                continue
            ax.quiver(e["gt_x"], e["gt_y"], e["ex"], e["ey"],
                      color=color, alpha=0.55, angles="xy", scale_units="xy",
                      scale=1, width=0.003, label=label)
        ax.set_title(KP_SHORT[kp], fontsize=9, pad=4)
        ax.invert_yaxis()   # image coords (y increases downward)
        ax.set_aspect("equal")
        ax.grid(zorder=0, alpha=0.3)
        ax.tick_params(labelsize=7)
        if ki == 0:
            ax.legend(fontsize=8, loc="upper left")

    fig.suptitle("Error Vectors — Arrow tail = GT, arrow head = model prediction\n"
                 f"(Red = N-model, Blue = L-model)", y=1.01)
    fig.tight_layout()
    fig_save(fig, pdf, "08_quiver_error_vectors")


# ═══════════════════════════════════════════════════════════════════════════
# PLOT 9 — Summary table (printed + saved as text image)
# ═══════════════════════════════════════════════════════════════════════════
def plot_summary_table(errors, rates, pdf):
    rows = []
    for kp in KEYPOINTS_8:
        n_e = errors["n"][kp]["dist"]
        l_e = errors["l"][kp]["dist"]
        rows.append({
            "Keypoint":    KP_SHORT[kp],
            "N  MAE (px)": f"{n_e.mean():.1f}" if len(n_e) else "N/A",
            "L  MAE (px)": f"{l_e.mean():.1f}" if len(l_e) else "N/A",
            "N  Std":      f"{n_e.std():.1f}"  if len(n_e) else "N/A",
            "L  Std":      f"{l_e.std():.1f}"  if len(l_e) else "N/A",
            "N  Det%":     f"{rates['n'][kp]:.0f}%",
            "L  Det%":     f"{rates['l'][kp]:.0f}%",
            "Winner":      "L-model" if (len(l_e) and len(n_e) and l_e.mean() < n_e.mean()) else ("N-model" if len(n_e) and len(l_e) else "—"),
        })

    df_table = pd.DataFrame(rows)
    print("\n" + "="*78)
    print("  SUMMARY TABLE")
    print("="*78)
    print(df_table.to_string(index=False))
    print("="*78)

    fig, ax = plt.subplots(figsize=(14, 4))
    fig.patch.set_facecolor(DARK_BG)
    ax.axis("off")

    cols   = list(df_table.columns)
    data   = df_table.values.tolist()
    tbl    = ax.table(cellText=data, colLabels=cols,
                      loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 2)

    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor(BORDER)
        if row == 0:
            cell.set_facecolor("#1e1e32")
            cell.set_text_props(color=ACCENT, fontweight="bold")
        else:
            val = data[row - 1][col] if col < len(data[row - 1]) else ""
            if col == len(cols) - 1:  # Winner column
                if "L-model" in str(val):
                    cell.set_facecolor("#0e1e2e")
                    cell.set_text_props(color=L_COLOR)
                elif "N-model" in str(val):
                    cell.set_facecolor("#2e1010")
                    cell.set_text_props(color=N_COLOR)
                else:
                    cell.set_facecolor(SURFACE)
            else:
                cell.set_facecolor(SURFACE)
                cell.set_text_props(color=TEXT)

    ax.set_title("Summary: N-model vs L-model vs Ground Truth", pad=20, color=TEXT)
    fig.tight_layout()
    fig_save(fig, pdf, "09_summary_table")

    # Also save summary CSV
    df_table.to_csv("output/comparison_summary.csv", index=False)
    print("  📁 comparison_summary.csv saved.")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main():
    # Validate inputs
    for path, name in [(NMODEL_CSV, "nmodel.csv"), (LMODEL_CSV, "lmodel.csv"), (GT_CSV, "groundtruth.csv")]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing: {path}\nRun extract_models.py and annotate_groundtruth.py first.")

    os.makedirs(OUT_DIR, exist_ok=True)

    print("📂 Loading CSVs …")
    merged = load_data()
    print(f"   Ground truth frames:  {len(merged)}")

    print("⚙️  Computing errors …")
    errors = compute_errors(merged)
    rates  = detection_rates(merged)

    # Quick sanity check
    total_n = sum(len(errors["n"][k]["dist"]) for k in KEYPOINTS_8)
    total_l = sum(len(errors["l"][k]["dist"]) for k in KEYPOINTS_8)
    print(f"   Valid N-model comparisons: {total_n}")
    print(f"   Valid L-model comparisons: {total_l}")

    if total_n == 0 and total_l == 0:
        print("⚠️  No overlapping frames between models and ground truth. "
              "Check that FRAME_STEP matches across all three scripts.")
        return

    print(f"\n🎨 Generating plots → {OUT_PDF}")
    with PdfPages(OUT_PDF) as pdf:
        plot_per_kp_mae(errors, pdf)
        plot_per_frame_mae(merged, errors, pdf)
        plot_scatter(errors, pdf)
        plot_heatmap(errors, merged, pdf, "n", "N-model", "04_heatmap_nmodel")
        plot_heatmap(errors, merged, pdf, "l", "L-model", "05_heatmap_lmodel")
        plot_boxplots(errors, pdf)
        plot_detection_rate(rates, pdf)
        plot_quiver(errors, pdf)
        plot_summary_table(errors, rates, pdf)

    print(f"\n✅ All done!")
    print(f"   PDF  : {OUT_PDF}")
    print(f"   PNGs : {OUT_DIR}/")


if __name__ == "__main__":
    main()
