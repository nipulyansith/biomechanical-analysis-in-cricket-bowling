import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

import pandas as pd


# ============================================================
# CONFIG: Update to match your metric script filenames
# ============================================================
SCRIPTS = [
    ("final_step_duration.py", "step_duration"),
    ("delivery_stride_metrics.py", "delivery_stride"),
    ("elbow_flexion_analysis.py", "elbow_flexion"),
    ("front_knee_flexion_analysis.py", "front_knee"),
    ("com.py", "head_com"),
    ("wrist_velocity.py", "wrist_velocity"),
]

FRAME_SUFFIX = "_frames.csv"
SUMMARY_SUFFIX = "_summary.json"


# ============================================================
# HELPERS
# ============================================================
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def flatten_json(obj, parent_key="", sep="__"):
    """
    Flatten nested dict/list JSON into a 1-level dict for CSV safety.
    Lists become indexed keys: key__0, key__1...
    """
    items = {}

    if isinstance(obj, dict):
        for k, v in obj.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
            items.update(flatten_json(v, new_key, sep=sep))

    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            new_key = f"{parent_key}{sep}{i}"
            items.update(flatten_json(v, new_key, sep=sep))

    else:
        items[parent_key] = obj

    return items


def clean_merge_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    If merge creates columns like col_new, we prefer the new values where not null,
    then drop *_new columns.
    """
    df = df.copy()
    new_cols = [c for c in df.columns if c.endswith("_new")]
    for c_new in new_cols:
        base = c_new[:-4]
        if base in df.columns:
            df[base] = df[c_new].combine_first(df[base])
        else:
            df[base] = df[c_new]
    if new_cols:
        df.drop(columns=new_cols, inplace=True)
    return df


def upsert_frame_dataset(
    trial_frames_path: Path,
    trial_id: str,
    video_name: str,
    fps: float,
    view_mode: str,
    cm_per_pixel,
    new_metric_frames: pd.DataFrame,
):
    """
    Merge new_metric_frames into per-trial frame dataset using 'frame' as key.
    Creates the file if missing.
    """

    if "frame" not in new_metric_frames.columns:
        raise ValueError("Metric frame CSV must contain a 'frame' column.")

    df_new = new_metric_frames.copy()
    df_new["frame"] = df_new["frame"].astype(int)

    base_cols = {
        "trial_id": trial_id,
        "video_name": video_name,
        "fps": float(fps),
        "view_mode": view_mode,
    }
    if cm_per_pixel is not None:
        base_cols["cm_per_pixel"] = float(cm_per_pixel)

    if trial_frames_path.exists():
        old = pd.read_csv(trial_frames_path)
        if "frame" not in old.columns:
            raise ValueError(f"{trial_frames_path} exists but has no 'frame' column.")
        old["frame"] = old["frame"].astype(int)

        merged = pd.merge(old, df_new, on="frame", how="outer", suffixes=("", "_new"))
        merged = clean_merge_columns(merged)

        # Fill metadata columns
        for k, v in base_cols.items():
            if k not in merged.columns:
                merged[k] = v
            else:
                merged[k] = merged[k].fillna(v)

        # Correct time axis: frame 1 = 0.0s
        if "time_s" not in merged.columns:
            merged["time_s"] = (merged["frame"] - 1) / merged["fps"]
        else:
            merged["time_s"] = merged["time_s"].fillna((merged["frame"] - 1) / merged["fps"])

        merged = merged.sort_values("frame").reset_index(drop=True)
        merged.to_csv(trial_frames_path, index=False)

    else:
        df = df_new.copy()
        for k, v in base_cols.items():
            df[k] = v

        df["time_s"] = (df["frame"] - 1) / df["fps"]
        df = df.sort_values("frame").reset_index(drop=True)
        df.to_csv(trial_frames_path, index=False)


def upsert_master_dataset(master_path: Path, trial_id: str, row_dict: dict):
    """
    Upsert one trial row in master_summary.csv (by trial_id).
    row_dict must be 1-level (flattened) values for safe CSV storage.
    """
    row = dict(row_dict)
    row["trial_id"] = trial_id

    if master_path.exists():
        master = pd.read_csv(master_path)
        if "trial_id" not in master.columns:
            raise ValueError(f"{master_path} exists but has no trial_id column.")

        # Add new columns if needed
        for k in row.keys():
            if k not in master.columns:
                master[k] = pd.NA

        if (master["trial_id"] == trial_id).any():
            idx = master.index[master["trial_id"] == trial_id][0]
            for k, v in row.items():
                master.at[idx, k] = v
        else:
            master = pd.concat([master, pd.DataFrame([row])], ignore_index=True)

        master.to_csv(master_path, index=False)

    else:
        pd.DataFrame([row]).to_csv(master_path, index=False)


def run_metric_script(
    script_path: str,
    metric_name: str,
    video_path: Path,
    trial_id: str,
    out_dir: Path,
    view_mode: str,
    bowling_arm: str,
):
    """
    Runs a metric script as subprocess.
    Contract:
      python <script> --video <path> --trial_id <id> --out_dir <dir>
                      --view_mode SIDE/FRONT --bowling_arm LEFT/RIGHT --metric_name <metric>
    Outputs expected:
      <out_dir>/<trial_id>/<metric_name>_frames.csv
      <out_dir>/<trial_id>/<metric_name>_summary.json
    """
    cmd = [
        sys.executable,
        script_path,
        "--video", str(video_path),
        "--trial_id", trial_id,
        "--out_dir", str(out_dir),
        "--view_mode", view_mode,
        "--bowling_arm", bowling_arm,
        "--metric_name", metric_name,
    ]

    print(f"\n▶ Running: {metric_name} ({script_path})")
    print("   " + " ".join(cmd))

    res = subprocess.run(cmd, capture_output=True, text=True)

    if res.returncode != 0:
        print("\n❌ Script failed:", metric_name)
        if res.stdout:
            print(res.stdout)
        if res.stderr:
            print(res.stderr)
        raise RuntimeError(f"{metric_name} failed with exit code {res.returncode}")

    if res.stdout.strip():
        print(res.stdout.strip())
    if res.stderr.strip():
        print("⚠️ stderr:", res.stderr.strip())


def load_metric_outputs(out_dir: Path, trial_id: str, metric_name: str):
    trial_folder = out_dir / trial_id
    frames_path = trial_folder / f"{metric_name}{FRAME_SUFFIX}"
    summary_path = trial_folder / f"{metric_name}{SUMMARY_SUFFIX}"

    if not frames_path.exists():
        raise FileNotFoundError(f"Missing frames output: {frames_path}")
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary output: {summary_path}")

    frames_df = pd.read_csv(frames_path)

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    if not isinstance(summary, dict):
        raise ValueError(f"Summary JSON must be a dict: {summary_path}")

    return frames_df, summary


def infer_trial_id(video_path: Path):
    return video_path.stem


# ============================================================
# MAIN PIPELINE
# ============================================================
def analyze_one_video(video_path: Path, trial_id: str, out_dir: Path, view_mode: str, bowling_arm: str):
    ensure_dir(out_dir)
    trial_dir = out_dir / trial_id
    ensure_dir(trial_dir)

    video_name = video_path.name

    trial_frames_path = trial_dir / f"{trial_id}_frames.csv"
    master_path = out_dir / "master_summary.csv"

    meta_summary = {
        "video_name": video_name,
        "view_mode": view_mode,
        "bowling_arm": bowling_arm,
        "processed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    fps = None
    cm_per_pixel = None

    for script_path, metric_name in SCRIPTS:
        if not Path(script_path).exists():
            raise FileNotFoundError(f"Script not found: {script_path}")

        run_metric_script(
            script_path=script_path,
            metric_name=metric_name,
            video_path=video_path,
            trial_id=trial_id,
            out_dir=out_dir,
            view_mode=view_mode,
            bowling_arm=bowling_arm,
        )

        frames_df, summary = load_metric_outputs(out_dir, trial_id, metric_name)

        # pick up fps/cm_per_pixel from any summary that has it
        if fps is None and summary.get("fps") is not None:
            fps = float(summary["fps"])
        if cm_per_pixel is None and summary.get("cm_per_pixel") is not None:
            cm_per_pixel = float(summary["cm_per_pixel"])

        # fallback fps if never provided yet
        use_fps = float(fps) if fps is not None else float(summary.get("fps", 30.0) or 30.0)

        upsert_frame_dataset(
            trial_frames_path=trial_frames_path,
            trial_id=trial_id,
            video_name=video_name,
            fps=use_fps,
            view_mode=view_mode,
            cm_per_pixel=cm_per_pixel,
            new_metric_frames=frames_df,
        )

        # flatten and prefix master keys to avoid collisions
        flat = flatten_json(summary)
        metric_prefixed = {}
        for k, v in flat.items():
            if k in ("trial_id", "video_name"):
                continue
            metric_prefixed[f"{metric_name}__{k}"] = v

        upsert_master_dataset(master_path, trial_id, metric_prefixed)

    # write meta
    upsert_master_dataset(master_path, trial_id, meta_summary)

    print("\n✅ DONE")
    print(f"📌 Trial frames dataset: {trial_frames_path}")
    print(f"📌 Master dataset:       {master_path}")
    print(f"📁 Trial folder:         {trial_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run all bowling analysis scripts and build datasets.")
    parser.add_argument("--video", type=str, default=None, help="Path to a single video file.")
    parser.add_argument("--input_dir", type=str, default=None, help="Folder containing videos (batch mode).")
    parser.add_argument("--out_dir", type=str, required=True, help="Output folder for datasets and per-trial results.")
    parser.add_argument("--trial_id", type=str, default=None, help="Trial ID (e.g., B01_T03). If not set, uses filename.")
    parser.add_argument("--view_mode", type=str, default="SIDE", choices=["SIDE", "FRONT"], help="Camera view mode.")
    parser.add_argument("--bowling_arm", type=str, default="RIGHT", choices=["LEFT", "RIGHT"], help="Bowling arm.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    if args.video is None and args.input_dir is None:
        raise ValueError("Provide either --video <file> or --input_dir <folder>")

    if args.video is not None:
        video_path = Path(args.video)
        if not video_path.exists():
            raise FileNotFoundError(video_path)
        trial_id = args.trial_id or infer_trial_id(video_path)
        analyze_one_video(video_path, trial_id, out_dir, args.view_mode, args.bowling_arm)
        return

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(input_dir)

    video_files = []
    for ext in ("*.mp4", "*.mov", "*.avi", "*.mkv", "*.MP4", "*.MOV", "*.AVI", "*.MKV"):
        video_files.extend(input_dir.glob(ext))
    video_files = sorted(video_files)

    if not video_files:
        print("No video files found in:", input_dir)
        return

    for vp in video_files:
        trial_id = infer_trial_id(vp)
        try:
            analyze_one_video(vp, trial_id, out_dir, args.view_mode, args.bowling_arm)
        except Exception as e:
            print(f"\n❌ Failed on {vp.name}: {e}")
            continue


if __name__ == "__main__":
    main()