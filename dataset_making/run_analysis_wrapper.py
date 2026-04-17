#!/usr/bin/env python3
"""
Wrapper script to run all.py with command-line parameters
This avoids the interactive prompts in all.py
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

def main():
    parser = argparse.ArgumentParser(description='Run bowling biomechanics analysis')
    parser.add_argument('--video', required=True, help='Path to video file')
    parser.add_argument('--hand', required=True, choices=['R', 'L'], help='Bowling hand (R or L)')
    parser.add_argument('--session-id', required=True, help='Session ID')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    parser.add_argument('--calib-top', required=True, help='Calibration top point (x,y)')
    parser.add_argument('--calib-bottom', required=True, help='Calibration bottom point (x,y)')
    parser.add_argument('--trial-id', default='', help='Optional trial ID for Excel files (defaults to video filename)')
    
    args = parser.parse_args()
    
    # Parse calibration points
    top_parts = args.calib_top.split(',')
    bottom_parts = args.calib_bottom.split(',')
    calib_top = (int(top_parts[0]), int(top_parts[1]))
    calib_bottom = (int(bottom_parts[0]), int(bottom_parts[1]))
    
    print(f"\n{'='*60}")
    print(f"Bowling Biomechanics Analysis Wrapper")
    print(f"{'='*60}")
    print(f"Video: {args.video}")
    print(f"Hand: {args.hand}")
    print(f"Calibration Top: {calib_top}")
    print(f"Calibration Bottom: {calib_bottom}")
    print(f"Output: {args.output_dir}")
    print(f"{'='*60}\n")
    
    # Now run the analysis directly
    try:
        import cv2
        from ultralytics import YOLO
        from scipy.signal import savgol_filter, find_peaks
        import numpy as np
        import pandas as pd
        
        # Import all.py functions
        print("Importing analysis modules...")
        import all as all_module
        from all import (
            extract_keypoints, detect_shared_events, 
            run_step_cadence, run_delivery_stride, run_elbow_flexion,
            run_knee_flexion, run_head_position, run_wrist_velocity,
            build_frame_dataset, write_frame_dataset,
            build_master_row, write_master_dataset,
            MODEL_PATH
        )
        
        # Setup directories
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # IMPORTANT: Set OUT_DIR in all.py so analysis functions save to correct location
        all_module.OUT_DIR = str(output_dir)
        print(f"Setting analysis output directory to: {all_module.OUT_DIR}")
        work_dir = output_dir / "work"
        work_dir.mkdir(exist_ok=True)
        
        # Load model
        print(f"\n📦 Loading YOLO model: {MODEL_PATH}")
        model = YOLO(MODEL_PATH)
        
        # Step 1: Extract keypoints
        print(f"\n🔍 Extracting keypoints from video...")
        df_xy, df_conf, fps, width, height, total_frames = extract_keypoints(args.video, model)
        print(f"✅ Extracted {len(df_xy)} frames at {fps} fps ({width}x{height})")
        
        # Step 2: Detect events
        print(f"\n📊 Detecting bowling events...")
        bowling_wrist = "right_wrist" if args.hand == "R" else "left_wrist"
        events = detect_shared_events(df_xy, fps, bowling_wrist, video_path=args.video)
        print(f"✅ Release frame: {events.get('release_frame', 'Unknown')}")
        
        # Step 3: Calculate calibration scale
        print(f"\n📏 Calculating calibration scale...")
        pixel_distance = abs(calib_bottom[1] - calib_top[1])
        stump_height = 0.711  # meters
        meters_per_pixel = stump_height / pixel_distance
        cm_per_pixel = meters_per_pixel * 100
        print(f"✅ Scale: {cm_per_pixel:.4f} cm/pixel")
        
        # Step 4: Run analyses
        print(f"\n⚙️  Running biomechanics analyses...")
        
        r_cadence = run_step_cadence(df_xy, fps, events, args.video, width, height)
        print(f"  ✅ Step cadence")
        
        r_stride = run_delivery_stride(df_xy, fps, events, meters_per_pixel, args.video, width, height)
        print(f"  ✅ Delivery stride")
        
        r_elbow = run_elbow_flexion(df_xy, fps, events, bowling_wrist, args.video, width, height)
        print(f"  ✅ Elbow flexion")
        
        knee_side = "L" if args.hand == "R" else "R"
        r_knee = run_knee_flexion(df_xy, df_conf, fps, events, knee_side, args.video, width, height)
        print(f"  ✅ Knee flexion")
        
        r_head = run_head_position(df_xy, fps, events, bowling_wrist, cm_per_pixel, meters_per_pixel, args.video, width, height)
        print(f"  ✅ Head position")
        
        r_wrist = run_wrist_velocity(df_xy, fps, events, bowling_wrist, meters_per_pixel, args.video, width, height)
        print(f"  ✅ Wrist velocity")
        
        # Step 5: Build and write Excel datasets
        print(f"\n📊 Building and writing Excel datasets...")
        
        # Get trial ID - use provided trial_id or default to video filename
        trial_id = args.trial_id if args.trial_id else Path(args.video).stem
        print(f"📝 Trial ID for Excel files: {trial_id}")
        
        # Pre-calculate bowling_arm_name like in all.py
        bowling_arm_name = "right" if args.hand == "R" else "left"
        
        # Determine front ankle and knee side for frame dataset
        kside = r_knee.get("kside") if r_knee and "kside" in r_knee else (
            "left" if knee_side == "L" else "right")
        front_ankle = (r_head.get("front_ankle") if r_head and "front_ankle" in r_head
                       else f"{events.get('f_side', 'left')}_ankle")
        
        frame_to_angle = r_elbow.get("frame_to_angle") if r_elbow else None
        frame_to_knee_angle = r_knee.get("frame_to_knee_angle") if r_knee else None
        
        # Build frame dataset
        df_frames = build_frame_dataset(
            trial_id=trial_id, 
            df_xy=df_xy, 
            fps=fps, 
            cm_per_pixel=cm_per_pixel,
            bowling_arm=bowling_arm_name,
            kside=kside, 
            front_ankle=front_ankle,
            frame_to_angle=frame_to_angle, 
            frame_to_knee_angle=frame_to_knee_angle,
        )
        
        # Write frame dataset to Excel file in dataset_making folder
        dataset_making_dir = Path(__file__).parent
        frames_excel_path = dataset_making_dir / "frames.xlsx"
        print(f"\n📊 Writing frame dataset...")
        print(f"   Path: {frames_excel_path}")
        print(f"   New rows: {len(df_frames)}")
        write_frame_dataset(df_frames, trial_id, str(frames_excel_path))
        print(f"✅ Frame dataset operation complete!")
        
        # Build master row
        master_row = build_master_row(
            trial_id=trial_id, 
            fps=fps, 
            hand=args.hand, 
            events=events,
            r_cadence=r_cadence, 
            r_stride=r_stride, 
            r_elbow=r_elbow,
            r_knee=r_knee, 
            r_head=r_head, 
            r_wrist=r_wrist,
        )
        
        # Write master dataset to Excel file in dataset_making folder
        master_excel_path = dataset_making_dir / "master.xlsx"
        print(f"\n📊 Writing master dataset...")
        print(f"   Path: {master_excel_path}")
        print(f"   New rows: 1")
        write_master_dataset(master_row, trial_id, str(master_excel_path))
        print(f"✅ Master dataset operation complete!")
        
        # Step 6: Save results - clean up dataframes before JSON serialization
        print(f"\n💾 Saving results...")
        
        # Clean results by removing non-JSON-serializable objects like DataFrames
        def clean_results(r):
            """Remove dataframes and other non-serializable objects"""
            if not isinstance(r, dict):
                return r
            cleaned = {}
            for k, v in r.items():
                if k in ('df_elbow', 'df_knee', 'df_xy', 'df_conf'):
                    # Skip dataframes - they'll be loaded from CSV
                    continue
                elif isinstance(v, dict):
                    cleaned[k] = clean_results(v)
                elif isinstance(v, (str, int, float, bool, type(None))):
                    cleaned[k] = v
                elif isinstance(v, (list, tuple)):
                    try:
                        cleaned[k] = [clean_results(item) if isinstance(item, dict) else item for item in v]
                    except:
                        pass  # Skip if can't process
            return cleaned
        
        # Ensure knee_extension is calculated correctly
        # Extension = Release - FFC (positive = straightening toward 180°, negative = bending from 180°)
        knee_at_ffc = r_knee.get('knee_angle_at_ffc') if r_knee else None
        knee_at_release = r_knee.get('knee_angle_at_release') if r_knee else None
        knee_extension = None
        if knee_at_ffc is not None and knee_at_release is not None:
            try:
                knee_extension = float(knee_at_release) - float(knee_at_ffc)
            except:
                pass
        
        r_knee_clean = clean_results(r_knee) if r_knee else {}
        if knee_extension is not None:
            r_knee_clean['knee_extension'] = knee_extension
        
        # Standardize head field names for frontend compatibility
        r_head_clean = clean_results(r_head) if r_head else {}
        if r_head_clean and 'ffc_offset' in r_head_clean:
            r_head_clean['head_offset_at_FFC'] = r_head_clean.pop('ffc_offset')
        if r_head_clean and 'bfc_offset' in r_head_clean:
            r_head_clean['head_offset_at_BFC'] = r_head_clean.pop('bfc_offset')
        
        # Standardize wrist field names for frontend compatibility
        r_wrist_clean = clean_results(r_wrist) if r_wrist else {}
        if r_wrist_clean:
            # Convert speed_at_release_mps to kmh for frontend
            if 'speed_at_release_mps' in r_wrist_clean and 'wrist_speed_at_release_kmh' not in r_wrist_clean:
                try:
                    r_wrist_clean['wrist_speed_at_release_kmh'] = float(r_wrist_clean['speed_at_release_mps']) * 3.6
                except:
                    pass
            # Create ball_speed_proxy structure if ball_proxy_kmh exists
            if 'ball_proxy_kmh' in r_wrist_clean:
                r_wrist_clean['ball_speed_proxy'] = {'kmh': r_wrist_clean['ball_proxy_kmh']}
        
        results_clean = {
            'cadence': clean_results(r_cadence),
            'stride': clean_results(r_stride),
            'elbow': clean_results(r_elbow),
            'knee': r_knee_clean,
            'head': r_head_clean,
            'wrist': r_wrist_clean
        }
        
        results_summary = {
            'session_id': args.session_id,
            'video': args.video,
            'bowling_hand': args.hand,
            'fps': fps,
            'total_frames': total_frames,
            'video_dimensions': {'width': width, 'height': height},
            'calibration': {
                'top': calib_top,
                'bottom': calib_bottom,
                'pixel_distance': pixel_distance,
                'meters_per_pixel': float(meters_per_pixel),
                'cm_per_pixel': float(cm_per_pixel)
            },
            'events': events,
            'results': results_clean
        }
        
        with open(output_dir / 'analysis_results.json', 'w') as f:
            json.dump(results_summary, f, indent=2, default=str)
        
        print(f"✅ Results saved to {output_dir / 'analysis_results.json'}")
        
        # Display current dataset status
        print(f"\n{'='*60}")
        print(f"📊 DISPLAYING ACCUMULATED DATASET STATUS")
        print(f"{'='*60}")
        all_module.display_dataset_status(dataset_making_dir)
        
        print(f"\n✨ Analysis completed successfully!")
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())
