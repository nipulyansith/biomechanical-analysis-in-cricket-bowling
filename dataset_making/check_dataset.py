#!/usr/bin/env python3
"""
Utility script to check and manage the accumulated dataset
Displays the current state of master.xlsx and frames.xlsx
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

def main():
    parser = argparse.ArgumentParser(description='Check accumulated dataset status')
    parser.add_argument('--remove-trial', type=str, help='Remove a specific trial from the dataset')
    parser.add_argument('--list-trials', action='store_true', help='List all trials in the dataset')
    parser.add_argument('--stats', action='store_true', help='Show detailed statistics')
    
    args = parser.parse_args()
    
    try:
        import all as all_module
        
        dataset_making_dir = Path(__file__).parent
        
        if args.remove_trial:
            print(f"\n🗑️  Removing trial: {args.remove_trial}")
            print("This feature is not yet implemented. Please manually edit the Excel files.")
            return 1
        
        if args.stats:
            print(f"\n📊 DETAILED DATASET STATISTICS")
            print(f"{'='*70}")
            
            # Read master dataset
            master_path = dataset_making_dir / "master.xlsx"
            if master_path.exists():
                df_master = all_module._safe_read_excel(str(master_path), all_module.MASTER_COLS)
                if len(df_master) > 0:
                    print(f"\n🎯 MASTER DATASET ({len(df_master)} trials)")
                    print(f"{'-'*70}")
                    for col in ['trial_id', 'bowling_arm', 'fps']:
                        if col in df_master.columns:
                            print(f"  {col}: {df_master[col].unique() if col == 'trial_id' else df_master[col].iloc[0] if len(df_master) > 0 else 'N/A'}")
                    
                    # Show summary statistics
                    print(f"\n  📈 Key Metrics Across Trials:")
                    numeric_cols = df_master.select_dtypes(include=['number']).columns
                    for col in ['step_duration_mean_s', 'stride_length_m', 'elbow_extension_deg']:
                        if col in numeric_cols:
                            try:
                                mean_val = df_master[col].mean()
                                std_val = df_master[col].std()
                                print(f"     {col}: {mean_val:.3f} ± {std_val:.3f}")
                            except:
                                pass
            
            # Read frames dataset  
            frames_path = dataset_making_dir / "frames.xlsx"
            if frames_path.exists():
                df_frames = all_module._safe_read_excel(str(frames_path), all_module.FRAME_COLS)
                if len(df_frames) > 0:
                    print(f"\n📋 FRAMES DATASET ({len(df_frames)} total frames)")
                    print(f"{'-'*70}")
                    for trial_id in sorted(df_frames['trial_id'].unique()):
                        count = len(df_frames[df_frames['trial_id'] == trial_id])
                        print(f"  {trial_id}: {count} frames")
        else:
            # Default: just display status
            all_module.display_dataset_status(dataset_making_dir)
            
            # Also show list of trials
            if args.list_trials:
                master_path = dataset_making_dir / "master.xlsx"
                if master_path.exists():
                    df_master = all_module._safe_read_excel(str(master_path), all_module.MASTER_COLS)
                    if len(df_master) > 0:
                        print(f"\n✅ Trials in dataset:")
                        for i, trial_id in enumerate(sorted(df_master['trial_id'].unique()), 1):
                            print(f"  {i}. {trial_id}")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())
