"""
Trial-Level Ground Truth vs Model Comparison
--------------------------------------------
Generates a suite of plots from a wide-format CSV where each metric has
three sub-columns: GT, Model, Error (GT-Model).

Trial IDs are auto-detected from the CSV. If they look like camera distances
(e.g. "10ft", "18ft", "30ft", "35ft"), the distance-comparison plot will
sort them numerically.

Outputs saved into ./plots/ :
    1. bland_altman.png         - agreement plots per metric (labeled points)
    2. scatter.png              - GT vs Model with identity + regression lines
    3. mae_rmse_bars.png        - MAE and RMSE per metric
    4. per_trial_error_bars.png - abs-error per trial, grouped by metric family
    5. error_heatmap.png        - signed error, metric x trial
    6. percent_error_heatmap.png - |error|/|GT| * 100, metric x trial
    7. distance_comparison.png  - error vs camera distance (sorted numerically)
    8. radar_accuracy.png       - normalized accuracy per metric, one polygon per trial

Usage:
    python analyze_trials.py [path/to/your.csv]
    # defaults to gt.csv in the current directory
"""

import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ----------------------------- CONFIG ----------------------------------------

# A palette to auto-assign colors to trials (extend if you have >8 trials)
PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
           "#9467bd", "#8c564b", "#e377c2", "#17becf"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]

# Rough grouping so we can plot related metrics together in per-trial bars
METRIC_GROUPS = {
    "Elbow":    ["Elbow Angle at Arm-Back", "Elbow Angle at Release",
                 "Elbow Extension (Arm-Back to Release)"],
    "Knee":     ["Front Knee Angle at FFC", "Front Knee Angle at Release"],
    "Timing":   ["Delivery Stride Duration", "Step Duration (Mean)",
                 "Step Duration (SD)", "Step Duration (CV)",
                 "Final 5 Steps Total Duration"],
    "Stride":   ["Delivery Stride Length"],
    "Head-FFC": ["Head COM Lateral Offset at FFC", "Head COM Vertical Offset at FFC",
                 "Head COM Distance at FFC"],
    "Head-BFC": ["Head COM Lateral Offset at BFC", "Head COM Vertical Offset at BFC",
                 "Head COM Distance at BFC"],
    "Speed":    ["Wrist Speed at Release"],
    "Frames":   ["Back Foot Contact (BFC) Frame", "Front Foot Contact (FFC) Frame",
                 "Arm-Back Event Frame", "Ball Release Frame"],
}

OUTDIR = Path("plots")
OUTDIR.mkdir(exist_ok=True)


# --------------------------- DATA LOADING ------------------------------------

def load_csv(path):
    """Read the two-header CSV. Returns (trials list, dict[metric -> DataFrame])."""
    raw = pd.read_csv(path, header=[1, 2])

    # Column 0 is Trial ID; grab it positionally then drop
    trials_series = raw.iloc[:, 0]
    raw = raw.iloc[:, 1:]
    mask = trials_series.notna()
    trials_series = trials_series[mask].reset_index(drop=True)
    raw = raw[mask].reset_index(drop=True)
    trials = trials_series.astype(str).tolist()

    metrics = {}
    for metric in raw.columns.get_level_values(0).unique():
        sub = raw[metric].copy()
        sub.columns = [c.strip() for c in sub.columns]
        for c in sub.columns:
            sub[c] = pd.to_numeric(sub[c], errors="coerce")
        sub.insert(0, "Trial", trials)
        metrics[metric] = sub.reset_index(drop=True)
    return trials, metrics


def parse_distance(trial_id):
    """Pull a numeric distance out of strings like '30ft', '10 ft', '30FT'.
    Returns float or None if not parseable."""
    m = re.search(r"(\d+(?:\.\d+)?)\s*ft", trial_id, flags=re.I)
    if m:
        return float(m.group(1))
    m = re.fullmatch(r"(\d+(?:\.\d+)?)", trial_id.strip())
    return float(m.group(1)) if m else None


def build_trial_style(trials):
    """Assign a color + marker to each trial. If all trials have a parseable
    distance, sort them by distance first so colors vary smoothly with range."""
    distances = [parse_distance(t) for t in trials]
    if all(d is not None for d in distances):
        order = sorted(range(len(trials)), key=lambda i: distances[i])
        ordered_trials = [trials[i] for i in order]
    else:
        ordered_trials = list(trials)

    colors, markers = {}, {}
    for i, t in enumerate(ordered_trials):
        colors[t] = PALETTE[i % len(PALETTE)]
        markers[t] = MARKERS[i % len(MARKERS)]
    return colors, markers


# ------------------------- PLOT HELPERS --------------------------------------

def _annotate_point(ax, x, y, label, color):
    ax.annotate(label, (x, y), textcoords="offset points",
                xytext=(6, 4), fontsize=7, color=color, alpha=0.9)


def _trial_legend(fig, trials, colors):
    handles = [Patch(facecolor=colors[t], label=t) for t in trials]
    fig.legend(handles=handles, loc="lower center", ncol=len(trials),
               frameon=False, bbox_to_anchor=(0.5, -0.01))


def _grid(n, cols=4):
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 3.4))
    axes = np.atleast_2d(axes).ravel()
    return fig, axes, rows, cols


# --------------------------- PLOTS -------------------------------------------

def plot_bland_altman(metrics, trials, colors, markers):
    names = list(metrics.keys())
    fig, axes, _, _ = _grid(len(names))
    for ax, name in zip(axes, names):
        df = metrics[name].dropna(subset=["GT", "Model"])
        if df.empty:
            ax.set_visible(False); continue
        diff = df["GT"] - df["Model"]
        bias = diff.mean()
        sd = diff.std(ddof=1) if len(diff) > 1 else 0.0
        loa_hi, loa_lo = bias + 1.96 * sd, bias - 1.96 * sd

        for _, row in df.iterrows():
            t = row["Trial"]
            x = (row["GT"] + row["Model"]) / 2
            y = row["GT"] - row["Model"]
            ax.scatter(x, y, color=colors.get(t, "gray"),
                       marker=markers.get(t, "o"), s=55, zorder=3)
            _annotate_point(ax, x, y, t, colors.get(t, "gray"))

        ax.axhline(bias, color="k", lw=1.2, label=f"Bias={bias:.2f}")
        ax.axhline(loa_hi, color="tab:red", ls="--", lw=1,
                   label=f"+1.96σ={loa_hi:.2f}")
        ax.axhline(loa_lo, color="tab:red", ls="--", lw=1,
                   label=f"-1.96σ={loa_lo:.2f}")
        ax.axhspan(loa_lo, loa_hi, color="tab:red", alpha=0.06)
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("Mean of GT & Model", fontsize=8)
        ax.set_ylabel("GT − Model", fontsize=8)
        ax.legend(fontsize=6, loc="best")
        ax.grid(alpha=0.3, ls=":")

    for ax in axes[len(names):]:
        ax.set_visible(False)
    fig.suptitle("Bland-Altman Agreement Plots (labeled by trial)",
                 fontsize=13, fontweight="bold")
    _trial_legend(fig, trials, colors)
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    fig.savefig(OUTDIR / "bland_altman.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_scatter(metrics, trials, colors, markers):
    names = list(metrics.keys())
    fig, axes, _, _ = _grid(len(names))
    for ax, name in zip(axes, names):
        df = metrics[name].dropna(subset=["GT", "Model"])
        if df.empty:
            ax.set_visible(False); continue
        gt, mo = df["GT"].values, df["Model"].values

        for _, row in df.iterrows():
            t = row["Trial"]
            ax.scatter(row["GT"], row["Model"],
                       color=colors.get(t, "gray"),
                       marker=markers.get(t, "o"), s=55, zorder=3)
            _annotate_point(ax, row["GT"], row["Model"], t, colors.get(t, "gray"))

        lo = float(min(gt.min(), mo.min()))
        hi = float(max(gt.max(), mo.max()))
        pad = (hi - lo) * 0.08 if hi > lo else 1
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
                "k--", lw=1, label="y = x")

        if len(gt) >= 2 and np.std(gt) > 0:
            slope, intercept = np.polyfit(gt, mo, 1)
            xs = np.linspace(lo - pad, hi + pad, 50)
            ax.plot(xs, slope * xs + intercept, lw=1.4,
                    color="tab:purple", label="fit")
            r = np.corrcoef(gt, mo)[0, 1]
        else:
            r = np.nan

        mae = np.mean(np.abs(gt - mo))
        rmse = np.sqrt(np.mean((gt - mo) ** 2))
        ax.text(0.03, 0.97, f"MAE={mae:.2f}\nRMSE={rmse:.2f}\nr={r:.3f}",
                transform=ax.transAxes, fontsize=7, va="top",
                bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.8))
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("Manual (GT)", fontsize=8)
        ax.set_ylabel("Model", fontsize=8)
        ax.legend(fontsize=6, loc="lower right")
        ax.grid(alpha=0.3, ls=":")

    for ax in axes[len(names):]:
        ax.set_visible(False)
    fig.suptitle("Ground Truth vs Model Scatter Plots",
                 fontsize=13, fontweight="bold")
    _trial_legend(fig, trials, colors)
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    fig.savefig(OUTDIR / "scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def compute_error_table(metrics):
    rows = {}
    for name, df in metrics.items():
        rows[name] = dict(zip(df["Trial"], df["Error (GT−Model)"]))
    return pd.DataFrame(rows).T


def plot_mae_rmse_bars(metrics):
    names, maes, rmses = [], [], []
    for n, df in metrics.items():
        err = df["Error (GT−Model)"].dropna().values
        if len(err) == 0:
            continue
        names.append(n)
        maes.append(np.mean(np.abs(err)))
        rmses.append(np.sqrt(np.mean(err ** 2)))

    order = np.argsort(maes)[::-1]
    names = [names[i] for i in order]
    maes = [maes[i] for i in order]
    rmses = [rmses[i] for i in order]

    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(names))))
    ax.barh(y - 0.2, maes, height=0.4, label="MAE", color="#4c72b0")
    ax.barh(y + 0.2, rmses, height=0.4, label="RMSE", color="#dd8452")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Error (metric-native units)")
    ax.set_title("Per-Metric Error Summary — MAE vs RMSE (sorted by MAE)",
                 fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(axis="x", alpha=0.3, ls=":")
    for i, (m, r) in enumerate(zip(maes, rmses)):
        ax.text(m, i - 0.2, f" {m:.2f}", va="center", fontsize=7)
        ax.text(r, i + 0.2, f" {r:.2f}", va="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(OUTDIR / "mae_rmse_bars.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_per_trial_error_bars(metrics, trials, colors):
    groups_to_plot = [(g, mets) for g, mets in METRIC_GROUPS.items()
                      if any(m in metrics for m in mets)]
    n = len(groups_to_plot)
    fig, axes, _, _ = _grid(n, cols=2)
    for ax, (gname, gmetrics) in zip(axes, groups_to_plot):
        gmetrics = [m for m in gmetrics if m in metrics]
        x = np.arange(len(gmetrics))
        width = 0.8 / max(len(trials), 1)
        for i, t in enumerate(trials):
            vals = []
            for m in gmetrics:
                df = metrics[m]
                row = df[df["Trial"] == t]
                v = row["Error (GT−Model)"].values
                vals.append(abs(v[0]) if len(v) and not np.isnan(v[0]) else 0)
            ax.bar(x + i * width - 0.4 + width / 2, vals, width=width,
                   color=colors.get(t, "gray"), label=t)
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace(" at ", "\n@") for m in gmetrics],
                           fontsize=7, rotation=0)
        ax.set_title(gname, fontsize=10, fontweight="bold")
        ax.set_ylabel("|Error|")
        ax.grid(axis="y", alpha=0.3, ls=":")
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.suptitle("Absolute Error per Trial, grouped by Metric Family",
                 fontsize=13, fontweight="bold")
    _trial_legend(fig, trials, colors)
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    fig.savefig(OUTDIR / "per_trial_error_bars.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _trial_column_order(trials):
    """If all trials parse to distances, return them sorted ascending;
    otherwise return as-given."""
    dists = [parse_distance(t) for t in trials]
    if all(d is not None for d in dists):
        return sorted(trials, key=lambda t: parse_distance(t))
    return list(trials)


def plot_error_heatmap(metrics, trials):
    col_order = _trial_column_order(trials)
    err_df = compute_error_table(metrics)[col_order]
    fig, ax = plt.subplots(figsize=(max(6, 0.9 * err_df.shape[1] + 2),
                                    max(5, 0.32 * err_df.shape[0])))
    vmax = np.nanmax(np.abs(err_df.values)) if err_df.size else 1
    im = ax.imshow(err_df.values, cmap="RdBu_r", aspect="auto",
                   vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(err_df.shape[1]))
    ax.set_xticklabels(err_df.columns, rotation=20, ha="right", fontsize=9)
    ax.set_yticks(range(err_df.shape[0]))
    ax.set_yticklabels(err_df.index, fontsize=8)
    for i in range(err_df.shape[0]):
        for j in range(err_df.shape[1]):
            v = err_df.values[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=6, color="black" if abs(v) < vmax * 0.6 else "white")
    ax.set_title("Signed Error (GT − Model) per Metric × Trial",
                 fontsize=12, fontweight="bold")
    fig.colorbar(im, ax=ax, label="Error (GT − Model)")
    fig.tight_layout()
    fig.savefig(OUTDIR / "error_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_percent_error_heatmap(metrics, trials):
    col_order = _trial_column_order(trials)
    rows = {}
    for name, df in metrics.items():
        pct = {}
        for _, r in df.iterrows():
            gt = r["GT"]; er = r["Error (GT−Model)"]
            if pd.isna(gt) or pd.isna(er) or gt == 0:
                pct[r["Trial"]] = np.nan
            else:
                pct[r["Trial"]] = abs(er) / abs(gt) * 100
        rows[name] = pct
    pct_df = pd.DataFrame(rows).T[col_order]

    fig, ax = plt.subplots(figsize=(max(6, 0.9 * pct_df.shape[1] + 2),
                                    max(5, 0.32 * pct_df.shape[0])))
    vmax = (min(100, np.nanpercentile(pct_df.values, 95))
            if pct_df.notna().any().any() else 100)
    im = ax.imshow(pct_df.values, cmap="Reds", aspect="auto", vmin=0, vmax=vmax)
    ax.set_xticks(range(pct_df.shape[1]))
    ax.set_xticklabels(pct_df.columns, rotation=20, ha="right", fontsize=9)
    ax.set_yticks(range(pct_df.shape[0]))
    ax.set_yticklabels(pct_df.index, fontsize=8)
    for i in range(pct_df.shape[0]):
        for j in range(pct_df.shape[1]):
            v = pct_df.values[i, j]
            if np.isnan(v):
                ax.text(j, i, "—", ha="center", va="center", fontsize=7, color="gray")
            else:
                ax.text(j, i, f"{v:.1f}%", ha="center", va="center",
                        fontsize=6,
                        color="white" if v > vmax * 0.55 else "black")
    ax.set_title(f"Percent Error |Err|/|GT|·100  (color capped at {vmax:.0f}%)",
                 fontsize=12, fontweight="bold")
    fig.colorbar(im, ax=ax, label="% error")
    fig.tight_layout()
    fig.savefig(OUTDIR / "percent_error_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_distance_comparison(metrics, trials):
    """For each metric, plot |error| vs numeric camera distance.
    Falls back to categorical x-axis if distances can't be parsed."""
    distances = {t: parse_distance(t) for t in trials}
    parseable = [t for t in trials if distances[t] is not None]

    if parseable:
        ordered = sorted(parseable, key=lambda t: distances[t])
        xs = [distances[t] for t in ordered]
        xlabel = "Camera distance (ft)"
        use_numeric = True
    else:
        ordered = list(trials)
        xs = list(range(len(trials)))
        xlabel = "Trial"
        use_numeric = False

    names = list(metrics.keys())
    fig, axes, _, _ = _grid(len(names))
    for ax, name in zip(axes, names):
        df = metrics[name]
        ys = []
        for t in ordered:
            row = df[df["Trial"] == t]
            v = row["Error (GT−Model)"].values
            ys.append(abs(v[0]) if len(v) and not np.isnan(v[0]) else np.nan)

        ax.plot(xs, ys, "-o", color="tab:blue", lw=1.5, ms=7)
        for x, y, t in zip(xs, ys, ordered):
            if not np.isnan(y):
                ax.annotate(f"{y:.2f}\n({t})", (x, y),
                            textcoords="offset points", xytext=(0, 8),
                            ha="center", fontsize=6)
        ax.axhline(0, color="gray", lw=0.6, ls=":")
        ax.set_title(name, fontsize=9)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel("|Error|", fontsize=8)
        if not use_numeric:
            ax.set_xticks(xs)
            ax.set_xticklabels(ordered, fontsize=7, rotation=15)
        ax.grid(alpha=0.3, ls=":")
    for ax in axes[len(names):]:
        ax.set_visible(False)
    title = ("Model Error vs Camera Distance" if use_numeric
             else "Model Error per Trial")
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    fig.savefig(OUTDIR / "distance_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_radar_accuracy(metrics, trials, colors, markers):
    err_df = compute_error_table(metrics).abs()
    max_abs = err_df.max(axis=1).replace(0, np.nan)
    norm = err_df.div(max_abs, axis=0).fillna(0)
    acc = (1 - norm).clip(lower=0)

    labels = list(acc.index)
    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    for t in trials:
        if t not in acc.columns:
            continue
        vals = acc[t].tolist() + [acc[t].iloc[0]]
        ax.plot(angles, vals, color=colors.get(t, "gray"),
                lw=1.8, label=t, marker=markers.get(t, "o"), ms=5)
        ax.fill(angles, vals, color=colors.get(t, "gray"), alpha=0.08)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([l.replace(" at ", "\n@") for l in labels], fontsize=7)
    ax.set_rlabel_position(0)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], fontsize=7)
    ax.set_ylim(0, 1)
    ax.set_title("Normalized Accuracy per Metric (outer = better)\n"
                 "1 − |error| / max|error_metric|",
                 fontsize=12, fontweight="bold", pad=20)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.08),
              ncol=len(trials), fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(OUTDIR / "radar_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ------------------------------ MAIN -----------------------------------------

def main(path):
    trials, metrics = load_csv(path)
    print(f"Loaded {len(trials)} trials, {len(metrics)} metrics")
    print("Trials:", trials)

    distances = {t: parse_distance(t) for t in trials}
    parseable = {t: d for t, d in distances.items() if d is not None}
    if parseable:
        print("Parsed distances:", parseable)

    colors, markers = build_trial_style(trials)

    plot_bland_altman(metrics, trials, colors, markers)
    plot_scatter(metrics, trials, colors, markers)
    plot_mae_rmse_bars(metrics)
    plot_per_trial_error_bars(metrics, trials, colors)
    plot_error_heatmap(metrics, trials)
    plot_percent_error_heatmap(metrics, trials)
    plot_distance_comparison(metrics, trials)
    plot_radar_accuracy(metrics, trials, colors, markers)

    print(f"\nAll plots saved to: {OUTDIR.resolve()}")
    for p in sorted(OUTDIR.glob("*.png")):
        print(" -", p.name)


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "gt.csv"
    main(csv_path)