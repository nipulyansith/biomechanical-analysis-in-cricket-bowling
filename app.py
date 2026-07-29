"""
FastBowl Lab - Backend Integration Server
==========================================
Integrates frontend UI with biomechanics analysis pipeline
"""

import os
import sys
import json
import uuid
import cv2
import subprocess
import shutil
from pathlib import Path

# Force UTF-8 output so emoji in print statements don't crash on Windows,
# where stdout defaults to the cp1252 codepage in some terminals/launchers.
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
from datetime import datetime
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import threading
import pandas as pd
import numpy as np

# ============================================================================
# Custom JSON Encoder to handle NaN and Infinity values
# ============================================================================
class SafeJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles NaN, Infinity, and -Infinity"""
    def encode(self, o):
        if isinstance(o, float):
            if np.isnan(o):
                return 'null'
            elif np.isinf(o):
                return 'null'
        return super().encode(o)
    
    def iterencode(self, o, _one_shot=False):
        """Encode the given object and yield each string representation as available."""
        for chunk in super().iterencode(o, _one_shot):
            yield chunk
    
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.integer, np.floating)):
            if np.isnan(o) or np.isinf(o):
                return None
            return float(o)
        return super().default(o)

def sanitize_for_json(obj):
    """Recursively replace NaN and Infinity values with None"""
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(item) for item in obj]
    elif isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    elif isinstance(obj, np.ndarray):
        return sanitize_for_json(obj.tolist())
    else:
        return obj

# ============================================================================
# Configuration
# ============================================================================
app = Flask(__name__)
app.json_encoder = SafeJSONEncoder  # Use custom encoder for NaN/Infinity handling
CORS(app)

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
RESULTS_DIR = BASE_DIR / "results"
DATASET_MAKING_DIR = BASE_DIR / "dataset_making"
FE_DIR = BASE_DIR / "FE"

UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi', 'mkv', 'webm', 'flv', 'wmv'}
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500 MB

# Session tracking
SESSIONS = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def resolve_unique_video_name(video_name_base):
    """
    Ensure video name is unique by adding a number suffix if needed.
    e.g., "My Video" becomes "My Video" if not exists, else "My Video 2", "My Video 3", etc.
    """
    if not video_name_base:
        return None
    
    # Clean the name
    video_name_base = str(video_name_base).strip()
    if not video_name_base:
        return None
    
    # Check if folder already exists
    result_path = RESULTS_DIR / video_name_base
    if not result_path.exists():
        return video_name_base
    
    # Find a unique name with a number suffix
    counter = 2
    while True:
        unique_name = f"{video_name_base} {counter}"
        result_path = RESULTS_DIR / unique_name
        if not result_path.exists():
            return unique_name
        counter += 1

def get_session_path(session_id):
    """Get the results path for a session, using result_folder if available"""
    if session_id in SESSIONS:
        result_folder = SESSIONS[session_id].get('result_folder', session_id)
    else:
        result_folder = session_id
    return RESULTS_DIR / result_folder

def create_session(video_name=None):
    """Create a new analysis session"""
    session_id = str(uuid.uuid4())[:8]
    
    # Resolve unique video name if provided
    unique_video_name = None
    if video_name:
        unique_video_name = resolve_unique_video_name(video_name)
    
    # Use video name as folder name if available, otherwise use session_id
    result_folder = unique_video_name if unique_video_name else session_id
    session_path = RESULTS_DIR / result_folder
    session_path.mkdir(exist_ok=True)
    
    SESSIONS[session_id] = {
        'id': session_id,
        'video_name': unique_video_name,
        'result_folder': result_folder,
        'created_at': datetime.now().isoformat(),
        'video_path': None,
        'bowling_arm': None,
        'calibration_top': None,
        'calibration_bottom': None,
        'status': 'created',
        'error': None,
        'progress': 0,
        'results': None
    }
    return session_id

# ============================================================================
# API Routes - Session Management
# ============================================================================

@app.route('/api/session/create', methods=['POST'])
def api_create_session():
    """Create new analysis session"""
    try:
        session_id = create_session()
        return jsonify({
            'success': True,
            'session_id': session_id,
            'message': 'Session created successfully'
        }), 201
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/session/<session_id>', methods=['GET'])
def api_get_session(session_id):
    """Get session details"""
    if session_id not in SESSIONS:
        return jsonify({'success': False, 'error': 'Session not found'}), 404
    
    session = SESSIONS[session_id].copy()
    return jsonify({'success': True, 'session': session}), 200

@app.route('/api/session/<session_id>', methods=['DELETE'])
def api_delete_session(session_id):
    """Delete session and cleanup"""
    if session_id not in SESSIONS:
        return jsonify({'success': False, 'error': 'Session not found'}), 404
    
    try:
        result_folder = SESSIONS[session_id].get('result_folder', session_id)
        session_path = RESULTS_DIR / result_folder
        if session_path.exists():
            shutil.rmtree(session_path)
        del SESSIONS[session_id]
        return jsonify({'success': True, 'message': 'Session deleted'}), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/sessions', methods=['GET'])
def api_list_sessions():
    """List all previous sessions with basic data"""
    try:
        sessions_list = []
        
        # Get sessions from memory
        for session_id, session in SESSIONS.items():
            # Use video_name for display if available, otherwise use result_folder
            display_name = session.get('video_name') or session.get('result_folder', session_id)
            session_data = {
                'session_id': session_id,
                'display_name': display_name,
                'created_at': session.get('created_at'),
                'status': session.get('status'),
                'bowling_arm': session.get('bowling_arm'),
                'results': session.get('results')
            }
            sessions_list.append(session_data)
        
        # Also load completed sessions from disk that aren't in memory
        if RESULTS_DIR.exists():
            for session_dir in RESULTS_DIR.iterdir():
                if session_dir.is_dir():
                    folder_name = session_dir.name
                    # Skip if already in memory
                    if any(s['result_folder'] == folder_name for s in [SESSIONS.get(sid) for sid in SESSIONS]):
                        continue
                    
                    # Try to load from disk
                    analysis_file = session_dir / "analysis_results.json"
                    if analysis_file.exists():
                        try:
                            with open(analysis_file) as f:
                                analysis_data = sanitize_for_json(json.load(f))
                            
                            # Extract bowling arm and results
                            results = {}
                            if 'results' in analysis_data:
                                r = analysis_data['results']
                                results = {
                                    'stride_duration': r.get('cadence'),
                                    'delivery_stride': r.get('stride'),
                                    'elbow_flexion': r.get('elbow'),
                                    'knee_flexion': r.get('knee'),
                                    'head_position': r.get('head'),
                                    'wrist_velocity': r.get('wrist')
                                }
                            
                            session_data = {
                                'session_id': folder_name,
                                'display_name': folder_name,
                                'created_at': analysis_data.get('timestamp', session_dir.stat().st_mtime),
                                'status': 'completed',
                                'bowling_arm': analysis_data.get('bowling_arm'),
                                'results': results
                            }
                            sessions_list.append(session_data)
                        except Exception as e:
                            print(f"Warning: Could not load session {session_id}: {str(e)}")
        
        # Sort by creation date (newest first)
        # Convert all created_at to timestamps for proper sorting
        def get_sort_key(session):
            created_at = session.get('created_at', '')
            if isinstance(created_at, str):
                try:
                    # Parse ISO format string to timestamp
                    from datetime import datetime
                    dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    return dt.timestamp()
                except:
                    return 0
            elif isinstance(created_at, (int, float)):
                return created_at
            else:
                return 0
        
        sessions_list.sort(key=get_sort_key, reverse=True)
        
        return jsonify({
            'success': True,
            'sessions': sessions_list,
            'count': len(sessions_list)
        }), 200
    except Exception as e:
        print(f"Error listing sessions: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

# ============================================================================
# API Routes - Video Upload
# ============================================================================

@app.route('/api/upload/<session_id>', methods=['POST'])
def api_upload_video(session_id):
    """Upload video file"""
    if session_id not in SESSIONS:
        return jsonify({'success': False, 'error': 'Session not found'}), 404
    
    if 'video' not in request.files:
        return jsonify({'success': False, 'error': 'No video file provided'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'success': False, 'error': 'Invalid file format'}), 400
    
    if file.content_length > MAX_FILE_SIZE:
        return jsonify({'success': False, 'error': 'File too large'}), 400
    
    try:
        # Check if video_name was provided in the upload
        video_name = request.form.get('video_name', '').strip() if 'video_name' in request.form else None
        
        # If video_name provided but session doesn't have one, update the result folder
        if video_name and not SESSIONS[session_id]['video_name']:
            unique_video_name = resolve_unique_video_name(video_name)
            if unique_video_name:
                # Move/create the new result folder
                old_folder = RESULTS_DIR / SESSIONS[session_id]['result_folder']
                new_folder = RESULTS_DIR / unique_video_name
                
                if old_folder.exists() and not new_folder.exists():
                    old_folder.rename(new_folder)
                    SESSIONS[session_id]['video_name'] = unique_video_name
                    SESSIONS[session_id]['result_folder'] = unique_video_name
                elif not new_folder.exists():
                    new_folder.mkdir(exist_ok=True)
                    SESSIONS[session_id]['video_name'] = unique_video_name
                    SESSIONS[session_id]['result_folder'] = unique_video_name
        
        # Save video
        filename = secure_filename(f"{session_id}_{datetime.now().timestamp()}.{file.filename.rsplit('.', 1)[1].lower()}")
        video_path = UPLOAD_DIR / filename
        file.save(str(video_path))
        
        # Verify video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            video_path.unlink()
            return jsonify({'success': False, 'error': 'Invalid video file'}), 400
        
        # Get video info
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        # Update session
        SESSIONS[session_id]['video_path'] = str(video_path)
        SESSIONS[session_id]['status'] = 'video_uploaded'
        
        return jsonify({
            'success': True,
            'message': 'Video uploaded successfully',
            'video_info': {
                'filename': filename,
                'fps': fps,
                'frame_count': frame_count,
                'width': width,
                'height': height,
                'duration_seconds': frame_count / fps if fps > 0 else 0
            }
        }), 200
    
    except Exception as e:
        if 'video_path' in locals() and video_path.exists():
            video_path.unlink()
        return jsonify({'success': False, 'error': f'Upload failed: {str(e)}'}), 500

# ============================================================================
# API Routes - Bowling Arm Selection
# ============================================================================

@app.route('/api/bowling-arm/<session_id>', methods=['POST'])
def api_set_bowling_arm(session_id):
    """Set bowling arm (R or L)"""
    if session_id not in SESSIONS:
        return jsonify({'success': False, 'error': 'Session not found'}), 404
    
    data = request.get_json()
    arm = data.get('arm', '').upper()
    
    if arm not in ['R', 'L']:
        return jsonify({'success': False, 'error': 'Arm must be R or L'}), 400
    
    SESSIONS[session_id]['bowling_arm'] = arm
    SESSIONS[session_id]['status'] = 'arm_selected'
    
    # Determine front knee
    front_knee = 'L' if arm == 'R' else 'R'
    
    return jsonify({
        'success': True,
        'message': f'Bowling arm set to {arm}',
        'front_knee': front_knee
    }), 200

# ============================================================================
# API Routes - Calibration
# ============================================================================

@app.route('/api/calibration/preview/<session_id>', methods=['GET'])
def api_calibration_preview(session_id):
    """Get first frame for calibration marking"""
    if session_id not in SESSIONS:
        return jsonify({'success': False, 'error': 'Session not found'}), 404
    
    session = SESSIONS[session_id]
    if not session['video_path']:
        return jsonify({'success': False, 'error': 'No video uploaded'}), 400
    
    try:
        cap = cv2.VideoCapture(session['video_path'])
        ok, frame = cap.read()
        cap.release()
        
        if not ok:
            return jsonify({'success': False, 'error': 'Could not read first frame'}), 500
        
        # Save preview image
        session_path = get_session_path(session_id)
        preview_path = session_path / 'calibration_preview.jpg'
        cv2.imwrite(str(preview_path), frame)
        
        return send_file(str(preview_path), mimetype='image/jpeg'), 200
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/calibration/<session_id>', methods=['POST'])
def api_set_calibration(session_id):
    """Set calibration points (top and bottom of stump)"""
    if session_id not in SESSIONS:
        return jsonify({'success': False, 'error': 'Session not found'}), 404
    
    data = request.get_json()
    
    if not data or 'top' not in data or 'bottom' not in data:
        return jsonify({'success': False, 'error': 'Missing calibration points'}), 400
    
    top = data.get('top')
    bottom = data.get('bottom')
    
    # Validate points format
    if not isinstance(top, dict) or 'x' not in top or 'y' not in top:
        return jsonify({'success': False, 'error': 'Invalid top point format'}), 400
    
    if not isinstance(bottom, dict) or 'x' not in bottom or 'y' not in bottom:
        return jsonify({'success': False, 'error': 'Invalid bottom point format'}), 400
    
    # Store calibration points
    SESSIONS[session_id]['calibration_top'] = (int(top['x']), int(top['y']))
    SESSIONS[session_id]['calibration_bottom'] = (int(bottom['x']), int(bottom['y']))
    SESSIONS[session_id]['status'] = 'calibrated'
    
    # Calculate calibration parameters
    pixels_per_meter = calculate_calibration_scale(
        top['y'], bottom['y'], stump_height=0.711
    )
    
    return jsonify({
        'success': True,
        'message': 'Calibration points set successfully',
        'calibration': {
            'top': {'x': top['x'], 'y': top['y']},
            'bottom': {'x': bottom['x'], 'y': bottom['y']},
            'pixels_per_meter': pixels_per_meter
        }
    }), 200

def calculate_calibration_scale(top_y, bottom_y, stump_height=0.711):
    """Calculate meters per pixel from calibration points"""
    pixel_distance = abs(bottom_y - top_y)
    if pixel_distance == 0:
        return 1.0
    meters_per_pixel = stump_height / pixel_distance
    return 1.0 / meters_per_pixel  # pixels per meter

# ============================================================================
# API Routes - Analysis Processing
# ============================================================================

@app.route('/api/process/<session_id>', methods=['POST'])
def api_process_video(session_id):
    """Start video processing"""
    if session_id not in SESSIONS:
        return jsonify({'success': False, 'error': 'Session not found'}), 404
    
    session = SESSIONS[session_id]
    
    # Validate session state
    if not session['video_path']:
        return jsonify({'success': False, 'error': 'No video uploaded'}), 400
    
    if not session['bowling_arm']:
        return jsonify({'success': False, 'error': 'Bowling arm not selected'}), 400
    
    if not session['calibration_top'] or not session['calibration_bottom']:
        return jsonify({'success': False, 'error': 'Calibration not set'}), 400
    
    # Start processing in background
    SESSIONS[session_id]['status'] = 'processing'
    SESSIONS[session_id]['progress'] = 0
    
    thread = threading.Thread(
        target=process_video_async,
        args=(session_id,)
    )
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'success': True,
        'message': 'Processing started',
        'session_id': session_id
    }), 202

def process_video_async(session_id):
    """Process video asynchronously"""
    session = SESSIONS.get(session_id)
    if not session:
        return
    
    try:
        print(f"\n{'='*60}")
        print(f"Starting processing for session: {session_id}")
        print(f"{'='*60}")
        
        session['progress'] = 10
        
        # Prepare parameters
        video_path = session['video_path']
        bowling_arm = session['bowling_arm']
        result_folder = session.get('result_folder', session_id)
        session_path = RESULTS_DIR / result_folder
        
        # Create working directory
        work_dir = session_path / "work"
        work_dir.mkdir(exist_ok=True)
        
        # Copy video to working directory
        work_video = work_dir / f"video.mp4"
        shutil.copy2(video_path, str(work_video))
        
        session['progress'] = 20
        
        print(f"🎬 Video copied to working directory: {work_video}")
        print(f"📊 Calibration Top: {session['calibration_top']}")
        print(f"📊 Calibration Bottom: {session['calibration_bottom']}")
        print(f"📁 Results folder: {result_folder}")
        
        # Run analysis with progress updates
        session['progress'] = 30
        
        run_analysis(
            session_id=session_id,
            video_path=str(work_video),
            bowling_arm=bowling_arm,
            calibration_top=session['calibration_top'],
            calibration_bottom=session['calibration_bottom'],
            output_dir=str(session_path)
        )
        
        session['progress'] = 85
        
        # Collect results
        print(f"📁 Collecting results from {session_path}")
        results = collect_results(session_id)
        
        session['progress'] = 100
        session['status'] = 'completed'
        session['results'] = results
        
        print(f"✅ Processing completed successfully!")
        
    except Exception as e:
        session['status'] = 'error'
        session['error'] = str(e)
        print(f"❌ Error processing session {session_id}: {str(e)}")
        import traceback
        traceback.print_exc()

def run_analysis(session_id, video_path, bowling_arm, calibration_top, 
                  calibration_bottom, output_dir):
    """Run the biomechanics analysis using wrapper script"""
    
    # Get the trial_id (video name / result folder name) from session
    session = SESSIONS.get(session_id)
    trial_id = session.get('result_folder', session_id) if session else session_id
    
    print(f"\n{'='*60}")
    print(f"Starting biomechanics analysis")
    print(f"{'='*60}")
    print(f"Session ID: {session_id}")
    print(f"Trial ID: {trial_id}")
    print(f"Video: {video_path}")
    print(f"Bowling arm: {bowling_arm}")
    print(f"Calibration top: {calibration_top}")
    print(f"Calibration bottom: {calibration_bottom}")
    print(f"Output dir: {output_dir}")
    
    # Use subprocess to run the wrapper script
    env = os.environ.copy()
    env['PYTHONPATH'] = str(DATASET_MAKING_DIR) + ':' + env.get('PYTHONPATH', '')
    env['PYTHONIOENCODING'] = 'utf-8'
    
    # Construct command
    cmd = [
        sys.executable,
        str(DATASET_MAKING_DIR / 'run_analysis_wrapper.py'),
        '--video', video_path,
        '--hand', bowling_arm,
        '--session-id', session_id,
        '--output-dir', output_dir,
        '--calib-top', f"{calibration_top[0]},{calibration_top[1]}",
        '--calib-bottom', f"{calibration_bottom[0]},{calibration_bottom[1]}",
        '--trial-id', trial_id
    ]
    
    print(f"\n🚀 Running command: {' '.join(cmd)}\n")
    
    # Run analysis
    result = subprocess.run(cmd, cwd=str(DATASET_MAKING_DIR), env=env,
                          capture_output=True, text=True, encoding='utf-8', timeout=600)
    
    # Print output for debugging
    if result.stdout:
        print("=== STDOUT ===")
        print(result.stdout)
    
    if result.stderr:
        print("=== STDERR ===")
        print(result.stderr)
    
    # Check result
    if result.returncode != 0:
        raise RuntimeError(f"Analysis failed with return code {result.returncode}: {result.stderr}")
    
    print(f"\n✅ Analysis completed successfully")

def collect_results(session_id):
    """Collect all results from output directory"""
    session = SESSIONS.get(session_id)
    result_folder = session.get('result_folder', session_id) if session else session_id
    session_path = RESULTS_DIR / result_folder
    results = {
        'stride_duration': None,
        'delivery_stride': None,
        'elbow_flexion': None,
        'knee_flexion': None,
        'head_position': None,
        'wrist_velocity': None,
        'keypoint_data': None,
        'videos': {},
        'raw_data': {}
    }
    
    # Try to find and load result files
    try:
        print(f"\n📁 Collecting results from {session_path}")
        
        # Check if analysis_results.json exists (summary from wrapper)
        summary_file = session_path / "analysis_results.json"
        if summary_file.exists():
            print(f"✅ Found analysis summary: {summary_file}")
            with open(summary_file) as f:
                summary = sanitize_for_json(json.load(f))  # Sanitize immediately after loading
                # Extract results from summary using correct key mappings
                if 'results' in summary:
                    r = summary['results']
                    # Map wrapper results keys to frontend expected keys
                    if r.get('cadence'): 
                        results['stride_duration'] = sanitize_for_json(r['cadence'])
                    if r.get('stride'): 
                        results['delivery_stride'] = sanitize_for_json(r['stride'])
                    if r.get('elbow'): 
                        results['elbow_flexion'] = sanitize_for_json(r['elbow'])
                    if r.get('knee'): 
                        knee_data = sanitize_for_json(r['knee'])
                        # Calculate knee_extension if not present
                        # Extension = Release - FFC (positive = straightening toward 180°)
                        if isinstance(knee_data, dict):
                            if 'knee_extension' not in knee_data and 'knee_angle_at_ffc' in knee_data and 'knee_angle_at_release' in knee_data:
                                try:
                                    ffc = float(knee_data['knee_angle_at_ffc'])
                                    release = float(knee_data['knee_angle_at_release'])
                                    knee_data['knee_extension'] = release - ffc
                                except:
                                    pass
                        results['knee_flexion'] = knee_data
                    if r.get('head'): 
                        head_data = sanitize_for_json(r['head'])
                        # Standardize field names: ffc_offset -> head_offset_at_FFC
                        if isinstance(head_data, dict):
                            if 'ffc_offset' in head_data and 'head_offset_at_FFC' not in head_data:
                                head_data['head_offset_at_FFC'] = head_data.pop('ffc_offset')
                            if 'bfc_offset' in head_data and 'head_offset_at_BFC' not in head_data:
                                head_data['head_offset_at_BFC'] = head_data.pop('bfc_offset')
                        results['head_position'] = head_data
                    if r.get('wrist'): 
                        wrist_data = sanitize_for_json(r['wrist'])
                        # Standardize and convert wrist fields
                        if isinstance(wrist_data, dict):
                            # Convert speed_at_release_mps to kmh if needed
                            if 'wrist_speed_at_release_kmh' not in wrist_data and 'speed_at_release_mps' in wrist_data:
                                try:
                                    wrist_data['wrist_speed_at_release_kmh'] = float(wrist_data['speed_at_release_mps']) * 3.6
                                except:
                                    pass
                            # Create ball_speed_proxy structure if needed
                            if 'ball_speed_proxy' not in wrist_data and 'ball_proxy_kmh' in wrist_data:
                                wrist_data['ball_speed_proxy'] = {'kmh': wrist_data['ball_proxy_kmh']}
                        results['wrist_velocity'] = wrist_data
                    
                    print(f"✅ Extracted results from analysis_results.json")
                    for key in ['stride_duration', 'delivery_stride', 'elbow_flexion', 'knee_flexion', 'head_position', 'wrist_velocity']:
                        if results[key]:
                            print(f"   ✓ {key}")
        
        # Backup: Also try loading individual files
        # Wrist velocity
        wrist_metrics = session_path / "wrist_ball_metrics.json"
        if wrist_metrics.exists() and not results['wrist_velocity']:
            print(f"✅ Found wrist metrics (backup): {wrist_metrics}")
            with open(wrist_metrics) as f:
                results['wrist_velocity'] = sanitize_for_json(json.load(f))
        
        # Knee flexion
        knee_csv = session_path / "knee_angles.csv"
        if knee_csv.exists() and not results['knee_flexion']:
            print(f"✅ Found knee angles (backup): {knee_csv}")
            df = pd.read_csv(knee_csv)
            results['knee_flexion'] = sanitize_for_json({
                'data': df.to_dict(orient='list'),
                'image': 'knee_flexion_analysis.png'
            })
        
        # Elbow flexion
        elbow_csv = session_path / "elbow_full_analysis.csv"
        if elbow_csv.exists() and not results['elbow_flexion']:
            print(f"✅ Found elbow analysis (backup): {elbow_csv}")
            df = pd.read_csv(elbow_csv)
            results['elbow_flexion'] = sanitize_for_json({
                'data': df.to_dict(orient='list'),
                'image': 'elbow_full_plots.png'
            })
        
        # Head position / COM
        head_metrics = session_path / "head_metrics.json"
        if head_metrics.exists() and not results['head_position']:
            print(f"✅ Found head metrics (backup): {head_metrics}")
            with open(head_metrics) as f:
                results['head_position'] = sanitize_for_json(json.load(f))
        
        # Keypoint data (skeletal visualization reference)
        keypoint_ref = session_path / "keypoint_samples" / "keypoint_reference.json"
        if keypoint_ref.exists():
            print(f"✅ Found keypoint reference data: {keypoint_ref}")
            with open(keypoint_ref) as f:
                keypoint_data = sanitize_for_json(json.load(f))
                results['keypoint_data'] = keypoint_data
        
        # Videos
        video_count = 0
        for video_file in session_path.glob("*_annotated.mp4"):
            results['videos'][video_file.stem.replace('_annotated', '')] = video_file.name
            video_count += 1
        
        if video_count > 0:
            print(f"✅ Found {video_count} annotated videos")
        
        print(f"✅ Results collection complete")
        print(f"\nFinal results structure:")
        for key, val in results.items():
            if key != 'videos' and key != 'raw_data' and key != 'keypoint_data':
                status = "✅" if val else "❌"
                print(f"  {status} {key}: {type(val).__name__}")
        
        if results.get('keypoint_data'):
            print(f"  ✅ keypoint_data: dict with {len(results['keypoint_data'].get('frames', []))} frames")
        
    except Exception as e:
        print(f"⚠️  Error collecting results: {str(e)}")
        import traceback
        traceback.print_exc()
    
    return results

@app.route('/api/progress/<session_id>', methods=['GET'])
def api_get_progress(session_id):
    """Get processing progress"""
    if session_id not in SESSIONS:
        return jsonify({'success': False, 'error': 'Session not found'}), 404
    
    session = SESSIONS[session_id]
    return jsonify({
        'success': True,
        'status': session['status'],
        'progress': session['progress'],
        'error': session['error']
    }), 200

# ============================================================================
# API Routes - Results Retrieval
# ============================================================================

@app.route('/api/results/<session_id>', methods=['GET'])
def api_get_results(session_id):
    """Get all analysis results"""
    # First check in-memory sessions
    if session_id in SESSIONS:
        session = SESSIONS[session_id]
        
        if session['status'] != 'completed':
            return jsonify({
                'success': False,
                'error': f'Analysis not completed. Status: {session["status"]}'
            }), 400
        
        return jsonify({
            'success': True,
            'results': sanitize_for_json(session['results'])
        }), 200
    
    # If not in memory, try loading from disk
    session_path = get_session_path(session_id)
    analysis_file = session_path / "analysis_results.json"
    
    if not analysis_file.exists():
        return jsonify({'success': False, 'error': 'Session not found'}), 404
    
    try:
        print(f"📂 Loading session {session_id} results from disk...")
        with open(analysis_file) as f:
            analysis_data = sanitize_for_json(json.load(f))
        
        # Extract results from analysis data
        results = {
            'stride_duration': None,
            'delivery_stride': None,
            'elbow_flexion': None,
            'knee_flexion': None,
            'head_position': None,
            'wrist_velocity': None,
            'videos': {},
            'raw_data': {}
        }
        
        if 'results' in analysis_data:
            r = analysis_data['results']
            if r.get('cadence'): results['stride_duration'] = r['cadence']
            if r.get('stride'): results['delivery_stride'] = r['stride']
            if r.get('elbow'): results['elbow_flexion'] = r['elbow']
            if r.get('knee'): results['knee_flexion'] = r['knee']
            if r.get('head'): results['head_position'] = r['head']
            if r.get('wrist'): results['wrist_velocity'] = r['wrist']
        
        print(f"✅ Loaded results from disk for session {session_id}")
        return jsonify({
            'success': True,
            'results': results
        }), 200
        
    except Exception as e:
        print(f"❌ Error loading results: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'Failed to load results: {str(e)}'}), 500

@app.route('/api/results/<session_id>/stride', methods=['GET'])
def api_get_stride_results(session_id):
    """Get step duration / stride results"""
    return api_get_analysis_file(session_id, 'step_cadence')

@app.route('/api/results/<session_id>/delivery', methods=['GET'])
def api_get_delivery_results(session_id):
    """Get delivery stride results"""
    return api_get_analysis_file(session_id, 'delivery_stride')

@app.route('/api/results/<session_id>/elbow', methods=['GET'])
def api_get_elbow_results(session_id):
    """Get elbow flexion results"""
    return api_get_analysis_file(session_id, 'elbow_flexion')

@app.route('/api/results/<session_id>/knee', methods=['GET'])
def api_get_knee_results(session_id):
    """Get knee flexion results"""
    return api_get_analysis_file(session_id, 'knee_flexion')

@app.route('/api/results/<session_id>/com', methods=['GET'])
def api_get_com_results(session_id):
    """Get center of mass / head position results"""
    return api_get_analysis_file(session_id, 'head_position')

@app.route('/api/results/<session_id>/wrist', methods=['GET'])
def api_get_wrist_results(session_id):
    """Get wrist velocity results"""
    return api_get_analysis_file(session_id, 'wrist_velocity')

def api_get_analysis_file(session_id, analysis_type):
    """Generic function to get analysis results"""
    if session_id not in SESSIONS:
        return jsonify({'success': False, 'error': 'Session not found'}), 404
    
    session = SESSIONS[session_id]
    
    if session['status'] != 'completed':
        return jsonify({
            'success': False,
            'error': f'Analysis not completed. Status: {session["status"]}'
        }), 400
    
    results = session.get('results', {})
    if analysis_type not in results:
        return jsonify({
            'success': False,
            'error': f'Results for {analysis_type} not available'
        }), 404
    
    return jsonify({
        'success': True,
        'data': results.get(analysis_type)
    }), 200

# ============================================================================
# API Routes - File Downloads
# ============================================================================

@app.route('/api/results/<session_id>/download/<file_type>', methods=['GET'])
def api_download_file(session_id, file_type):
    """Download result files (CSVs, images, videos)"""
    if session_id not in SESSIONS:
        return jsonify({'success': False, 'error': 'Session not found'}), 404
    
    session_path = get_session_path(session_id)
    
    file_map = {
        'elbow_csv': 'elbow_full_analysis.csv',
        'elbow_png': 'elbow_full_plots.png',
        'knee_csv': 'knee_angles.csv',
        'knee_png': 'knee_flexion_analysis.png',
        'wrist_csv': 'wrist_timeseries.csv',
        'wrist_png': 'wrist_speed_vs_time.png',
        'head_csv': 'head_metrics.csv',
        'stride_video': 'step_cadence_annotated.mp4',
        'delivery_video': 'delivery_stride_annotated.mp4',
        'elbow_video': 'elbow_flexion_annotated.mp4',
        'knee_video': 'knee_flexion_annotated.mp4',
        'head_video': 'head_position_annotated.mp4',
    }
    
    if file_type not in file_map:
        return jsonify({'success': False, 'error': 'Invalid file type'}), 400
    
    file_path = session_path / file_map[file_type]
    
    if not file_path.exists():
        return jsonify({'success': False, 'error': 'File not found'}), 404
    
    return send_file(str(file_path), download_name=file_path.name), 200

@app.route('/api/results/<session_id>/image/<image_name>', methods=['GET'])
def api_get_image(session_id, image_name):
    """Get analysis image"""
    session_path = get_session_path(session_id)
    image_path = session_path / image_name
    
    if not image_path.exists():
        return jsonify({'success': False, 'error': 'Image not found'}), 404
    
    return send_file(str(image_path), mimetype='image/png'), 200

@app.route('/api/results/<session_id>/video/<video_name>', methods=['GET'])
def api_get_video(session_id, video_name):
    """Get analysis video with streaming and range request support"""
    from flask import request
    
    session_path = RESULTS_DIR / session_id
    video_path = session_path / video_name
    
    if not video_path.exists():
        return jsonify({'success': False, 'error': 'Video not found'}), 404
    
    file_size = video_path.stat().st_size
    
    # Handle range requests for streaming
    range_header = request.headers.get('Range')
    if range_header:
        # Parse the range header (e.g., "bytes=0-1023")
        try:
            range_match = range_header.replace('bytes=', '').split('-')
            start = int(range_match[0]) if range_match[0] else 0
            end = int(range_match[1]) if range_match[1] else file_size - 1
            
            # Clamp values
            start = max(0, start)
            end = min(file_size - 1, end)
            
            # Return 206 Partial Content
            with open(str(video_path), 'rb') as f:
                f.seek(start)
                data = f.read(end - start + 1)
            
            response = app.response_class(
                data,
                206,  # 206 Partial Content
                mimetype='video/mp4',
                direct_passthrough=True
            )
            response.headers['Content-Range'] = f'bytes {start}-{end}/{file_size}'
            response.headers['Content-Length'] = end - start + 1
            response.headers['Accept-Ranges'] = 'bytes'
            response.headers['Content-Type'] = 'video/mp4'
            response.headers['Cache-Control'] = 'public, max-age=86400'
            return response
        except (ValueError, IndexError):
            pass
    
    # Return full file if no range header
    def generate():
        with open(str(video_path), 'rb') as f:
            while True:
                chunk = f.read(8192)  # 8KB chunks
                if not chunk:
                    break
                yield chunk
    
    response = app.response_class(generate(), 200, mimetype='video/mp4')
    response.headers['Accept-Ranges'] = 'bytes'
    response.headers['Content-Type'] = 'video/mp4'
    response.headers['Content-Length'] = file_size
    response.headers['Cache-Control'] = 'public, max-age=86400'
    response.headers['Content-Disposition'] = 'inline; filename=' + video_name
    return response

@app.route('/api/results/<session_id>/keypoints', methods=['GET'])
def api_get_keypoints(session_id):
    """Get keypoint reference data"""
    session_path = RESULTS_DIR / session_id
    keypoint_ref = session_path / "keypoint_samples" / "keypoint_reference.json"
    
    if not keypoint_ref.exists():
        return jsonify({'success': False, 'error': 'Keypoint data not found'}), 404
    
    try:
        with open(keypoint_ref) as f:
            keypoint_data = sanitize_for_json(json.load(f))
        
        return jsonify({
            'success': True,
            'data': keypoint_data
        }), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/results/<session_id>/keypoint-frame/<frame_label>', methods=['GET'])
def api_get_keypoint_frame(session_id, frame_label):
    """Get annotated keypoint frame image"""
    session_path = get_session_path(session_id)
    keypoint_samples_dir = session_path / "keypoint_samples"
    
    # Find the frame file matching the label
    if not keypoint_samples_dir.exists():
        return jsonify({'success': False, 'error': 'Keypoint samples not found'}), 404
    
    # Search for frame file with the given label
    for frame_file in keypoint_samples_dir.glob(f"keypoints_{frame_label}_frame_*.jpg"):
        return send_file(str(frame_file), mimetype='image/jpeg'), 200
    
    return jsonify({'success': False, 'error': f'Frame {frame_label} not found'}), 404

# ============================================================================
# Frontend Routes
# ============================================================================

@app.route('/', methods=['GET'])
def serve_index():
    """Serve dashboard"""
    return send_from_directory(str(FE_DIR), 'dashboard.html')

@app.route('/<path:filepath>', methods=['GET'])
def serve_static(filepath):
    """Serve static files"""
    return send_from_directory(str(FE_DIR), filepath)

# ============================================================================
# Health Check
# ============================================================================

@app.route('/api/health', methods=['GET'])
def api_health():
    """Health check endpoint"""
    return jsonify({
        'success': True,
        'status': 'healthy',
        'timestamp': datetime.now().isoformat()
    }), 200

# ============================================================================
# Error Handlers
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'success': False, 'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'success': False, 'error': 'Internal server error'}), 500

# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("  FastBowl Lab - Backend Server")
    print("="*60)
    print(f"Upload directory: {UPLOAD_DIR}")
    print(f"Results directory: {RESULTS_DIR}")
    print(f"FE directory: {FE_DIR}")
    print("\nServer starting on http://localhost:5001")
    print("="*60 + "\n")
    
    app.run(debug=False, host='0.0.0.0', port=5001, threaded=True)
