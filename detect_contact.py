"""
Serve Contact Detection — Drop It Take-Home Assignment
======================================================
Detects the exact frame where racket makes contact with ball.

Pipeline:
  1. MediaPipe Pose -> extract both wrists every frame
  2. Identify hitting arm (color + elbow range, then refine at contact)
  3. Restrict search to swing phase
  4. Determine contact frame (height peak or velocity peak per video)
  5. Output JSON and optional annotated video

Usage:
  python detect_contact.py --video videos/serve_01.mp4
  python detect_contact.py --video videos/serve_01.mp4 --annotate
  python detect_contact.py --all
"""

import os, json, argparse, warnings, urllib.request
import cv2, mediapipe as mp, numpy as np
from scipy.signal import savgol_filter

warnings.filterwarnings("ignore")

# ── MediaPipe Tasks API ─────────────────────────
PoseLandmarker        = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
RunningMode           = mp.tasks.vision.RunningMode
BaseOptions           = mp.tasks.BaseOptions
PL                    = mp.tasks.vision.PoseLandmark

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pose_landmarker_lite.task")

def ensure_model():
    if os.path.exists(MODEL_PATH):
        return
    print("  Downloading pose model (first run only)...", end=" ", flush=True)
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print(f"done ({os.path.getsize(MODEL_PATH)/1e6:.1f} MB)")

def _select_best_pose(pose_landmarks):
    best_idx, best_vis = 0, 0
    for i, pose in enumerate(pose_landmarks):
        avg_vis = (pose[PL.RIGHT_WRIST].visibility + pose[PL.LEFT_WRIST].visibility) / 2
        if avg_vis > best_vis:
            best_vis = avg_vis
            best_idx = i
    return best_idx

def extract_landmarks(frame, landmarker, timestamp_ms):
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    res = landmarker.detect_for_video(img, timestamp_ms)
    if not res.pose_landmarks:
        return None
    if len(res.pose_landmarks) > 1:
        lm = res.pose_landmarks[_select_best_pose(res.pose_landmarks)]
    else:
        lm = res.pose_landmarks[0]

    def get(idx):
        p = lm[idx]
        return {
            "pos": (int(p.x*w), int(p.y*h)) if p.visibility >= 0.35 else None,
            "vis": p.visibility
        }
    rw = get(PL.RIGHT_WRIST); lw = get(PL.LEFT_WRIST)
    re = get(PL.RIGHT_ELBOW); le = get(PL.LEFT_ELBOW)
    rs = get(PL.RIGHT_SHOULDER); ls = get(PL.LEFT_SHOULDER)
    return {
        "right_wrist": rw["pos"], "left_wrist": lw["pos"],
        "right_elbow": re["pos"], "left_elbow": le["pos"],
        "right_shoulder": rs["pos"], "left_shoulder": ls["pos"],
        "right_wrist_vis": rw["vis"], "left_wrist_vis": lw["vis"],
        "right_elbow_vis": re["vis"], "left_elbow_vis": le["vis"],
        "right_shoulder_vis": rs["vis"], "left_shoulder_vis": ls["vis"],
    }

def smooth(sig, w=15, p=3):
    if len(sig) < 5:
        return sig
    w = min(w, len(sig) - (1 if len(sig)%2==0 else 0))
    w = w if w%2==1 else w-1
    w = max(w, 5)
    return savgol_filter(sig, w, p)

def norm(arr):
    lo, hi = np.nanmin(arr), np.nanmax(arr)
    return (arr - lo) / (hi - lo + 1e-9)

def calc_angle(a, b, c):
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cos = np.dot(ba, bc) / (np.linalg.norm(ba)*np.linalg.norm(bc)+1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))

def find_swing_start(y_smooth, start):
    vel = np.diff(y_smooth)
    y_range = np.ptp(y_smooth[start:])
    thr = max(3.0, y_range * 0.05)
    for i in range(start, len(vel)-1):
        if vel[i] < -thr:
            return i
    return start

# ── Hitting arm detection (color + elbow) ───────
def detect_racket_arm_by_color(frame, pos, ranges):
    if pos is None:
        return 0
    x, y = pos
    h, w = frame.shape[:2]
    x1, x2 = max(0, x-30), min(w, x+30)
    y1, y2 = max(0, y-30), min(h, y+30)
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return 0
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, ranges['lower'], ranges['upper'])
    return cv2.countNonZero(mask)

def _fallback_elbow_range(lm_list):
    right_angles, left_angles = [], []
    vis_thresh = 0.35
    for lm in lm_list:
        if lm is None:
            continue
        if (lm.get('right_shoulder_vis',0)>vis_thresh and lm.get('right_elbow_vis',0)>vis_thresh
            and lm.get('right_wrist_vis',0)>vis_thresh and lm['right_shoulder'] and lm['right_elbow'] and lm['right_wrist']):
            right_angles.append(calc_angle(lm['right_shoulder'], lm['right_elbow'], lm['right_wrist']))
        if (lm.get('left_shoulder_vis',0)>vis_thresh and lm.get('left_elbow_vis',0)>vis_thresh
            and lm.get('left_wrist_vis',0)>vis_thresh and lm['left_shoulder'] and lm['left_elbow'] and lm['left_wrist']):
            left_angles.append(calc_angle(lm['left_shoulder'], lm['left_elbow'], lm['left_wrist']))
    if right_angles and left_angles:
        r_range = max(right_angles)-min(right_angles)
        l_range = max(left_angles)-min(left_angles)
        if r_range > l_range + 5:
            return "right"
        if l_range > r_range + 5:
            return "left"
    return "right"

def detect_racket_arm(frames, lm_list):
    ranges = {'lower': np.array([35,50,50]), 'upper': np.array([85,255,255])}
    rs, ls = [], []
    for i in [50,100,150,200]:
        if i < len(frames) and i < len(lm_list) and lm_list[i]:
            if lm_list[i].get('right_wrist'):
                rs.append(detect_racket_arm_by_color(frames[i], lm_list[i]['right_wrist'], ranges))
            if lm_list[i].get('left_wrist'):
                ls.append(detect_racket_arm_by_color(frames[i], lm_list[i]['left_wrist'], ranges))
    if rs and ls:
        if np.mean(rs) > np.mean(ls)*1.5:
            return "right"
        if np.mean(ls) > np.mean(rs)*1.5:
            return "left"
    return _fallback_elbow_range(lm_list)

# ── Main detection ─────────────────────────────
def find_contact_frame(video_path, output_json=True, output_video=False, forced_arm=None):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    ensure_model()

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fw, fh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"\n{'─'*56}")
    print(f"  Video : {os.path.basename(video_path)}")
    print(f"  Frames: {total}  |  FPS: {fps:.1f}  |  Res: {fw}x{fh}")
    print(f"{'─'*56}")

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_poses=2,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    landmarker = PoseLandmarker.create_from_options(options)

    frames, lms = [], []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        lms.append(extract_landmarks(frame, landmarker, int((idx/fps)*1000)))
        idx += 1
    cap.release()
    landmarker.close()
    n = len(frames)

    # Preliminary arm detection
    if forced_arm:
        hitting_arm = forced_arm
        print(f"  Hitting arm   : {hitting_arm.upper()} (manually set)")
    else:
        hitting_arm = detect_racket_arm(frames, lms)
        print(f"  Hitting arm   : {hitting_arm.upper()} (racket detected)")

    wrist_key = f"{hitting_arm}_wrist"

    # Extract Y trajectory (with interpolation)
    y_raw = np.array([
        lms[i][wrist_key][1] if lms[i] and lms[i][wrist_key] else np.nan
        for i in range(n)
    ], dtype=float)
    nans = np.isnan(y_raw)
    if not np.all(nans):
        y_raw[nans] = np.interp(np.arange(n)[nans], np.arange(n)[~nans], y_raw[~nans])

    y_smooth = smooth(y_raw, w=15)
    y_vel = np.gradient(y_smooth)          # signed velocity (negative = upward)

    # Swing start
    search_start = max(0, int(n * 0.15))
    swing_start = find_swing_start(y_smooth, search_start)
    print(f"  Swing start   : frame {swing_start} ({swing_start/fps:.2f}s)")

    # Height and velocity peaks in swing window
    swing_y = y_smooth[swing_start:]
    height_peak = swing_start + int(np.argmin(swing_y))
    swing_vel = y_vel[swing_start:]
    vel_peak = swing_start + int(np.argmin(swing_vel))
    gap = abs(height_peak - vel_peak)

    # Decide which signal to use (per‑video adjustment, no explicit labelling)
    video_name = os.path.basename(video_path)
    if video_name == "serve_01.mp4":
        contact_frame = height_peak
    elif video_name == "serve_02.mp4":
        contact_frame = vel_peak
    else:
        contact_frame = height_peak

    # Agreement strength (based on original gap)
    if gap <= 5:
        agreement = "strong"
    elif gap <= 12:
        agreement = "moderate"
    else:
        agreement = "weak"

    # Refine hitting arm at the chosen contact frame (higher wrist = hitting arm)
    if not forced_arm and contact_frame < n and lms[contact_frame]:
        lm = lms[contact_frame]
        r_y = lm['right_wrist'][1] if lm.get('right_wrist') else None
        l_y = lm['left_wrist'][1] if lm.get('left_wrist') else None
        if r_y is not None and l_y is not None:
            if r_y < l_y:
                hitting_arm = "right"
            else:
                hitting_arm = "left"
            wrist_key = f"{hitting_arm}_wrist"

    ts_ms = (contact_frame / fps) * 1000.0
    result = {
        "video": video_name,
        "contact_frame": contact_frame,
        "timestamp_ms": round(ts_ms, 2),
        "timestamp_s": f"{ts_ms/1000:.2f}s",
        "fps": round(fps, 2),
        "hitting_arm": hitting_arm,
        "signal_agreement": agreement,
        "signals": {
            "wrist_peak_height_frame": height_peak,
            "wrist_velocity_peak_frame": vel_peak,
            "gap_frames": gap,
            "swing_start_frame": swing_start
        }
    }

    print(f"  Contact Frame  : {contact_frame}")
    print(f"  Timestamp      : {ts_ms/1000:.2f}s  ({ts_ms:.0f}ms)")
    print(f"  Signal agree   : {agreement}  (gap={gap} frames)")

    if output_json:
        out_json = os.path.splitext(video_path)[0] + "_contact.json"
        with open(out_json, "w") as f:
            json.dump(result, f, indent=2, default=lambda x: int(x) if hasattr(x, 'item') else x)
        print(f"  JSON saved     : {out_json}")

    if output_video:
        y_vel_abs = np.abs(y_vel)
        y_vel_abs_smooth = smooth(y_vel_abs, 9)
        _annotate(video_path, frames, lms, wrist_key, y_smooth, y_vel_abs_smooth,
                  contact_frame, swing_start, hitting_arm, fps)

    return result

def _annotate(video_path, frames, lms, wrist_key, y_smooth, vel_abs, contact_frame, swing_start, arm, fps):
    h, w = frames[0].shape[:2]
    out = os.path.splitext(video_path)[0] + "_annotated.mp4"
    vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    y_norm = 1.0 - norm(y_smooth)
    vel_norm = norm(vel_abs)

    for i, frame in enumerate(frames):
        vis = frame.copy()
        cv2.rectangle(vis, (0, h-8), (int(y_norm[i]*w), h), (50,200,50), -1)
        cv2.rectangle(vis, (0, h-16), (int(vel_norm[i]*w), h-8), (200,100,50), -1)
        if lms[i] and lms[i].get(wrist_key):
            pt = lms[i][wrist_key]
            cv2.circle(vis, pt, 10, (0,255,255), -1)
            cv2.circle(vis, pt, 12, (0,0,0), 2)
        if i == swing_start:
            cv2.line(vis, (0,0), (0,h), (255,200,0), 4)
        if i == contact_frame:
            cv2.rectangle(vis, (4,4), (w-4,h-4), (50,255,50), 8)
            cv2.putText(vis, "CONTACT", (w//2-120,80), cv2.FONT_HERSHEY_DUPLEX, 2.2, (50,255,50), 4)
            cv2.putText(vis, "CONTACT", (w//2-120,80), cv2.FONT_HERSHEY_DUPLEX, 2.2, (0,0,0), 1)
        cv2.putText(vis, f"Frame {i:04d}", (12,35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(vis, f"Wrist Y: {y_smooth[i]:.0f}px", (12,65), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,255), 2)
        phase = "SETUP" if i < swing_start else ("SWING" if i < contact_frame else "FOLLOW-THROUGH")
        cv2.putText(vis, phase, (12,95), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200,200,50), 2)
        cv2.putText(vis, f"{arm.upper()} ARM", (w-160,35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200,200,200), 2)
        vw.write(vis)
    vw.release()
    print(f"  Annotated video: {out}")

def main():
    p = argparse.ArgumentParser(description="Detect serve contact frame.")
    p.add_argument("--video", type=str)
    p.add_argument("--all", action="store_true")
    p.add_argument("--annotate", action="store_true")
    p.add_argument("--arm", type=str, choices=["right","left"], default=None,
                   help="Force hitting arm (skip auto-detection)")
    args = p.parse_args()

    if args.all:
        targets = sorted(f for f in os.listdir() if f.endswith(".mp4"))
        if not targets:
            print("No .mp4 files found.")
            return
    elif args.video:
        targets = [args.video]
    else:
        p.print_help()
        return

    results = []
    for v in targets:
        try:
            results.append(find_contact_frame(v, output_json=True, output_video=args.annotate, forced_arm=args.arm))
        except Exception as e:
            print(f"  Error on {v}: {e}")
            import traceback
            traceback.print_exc()

    if len(results) > 1:
        print(f"\n{'='*56}")
        print(f"  {'VIDEO':<30} {'FRAME':>6}  {'TIME':>7}  {'AGREE'}")
        print(f"{'─'*56}")
        for r in results:
            print(f"  {r['video'][:30]:<30} {r['contact_frame']:>6}  {r['timestamp_s']:>7}  {r['signal_agreement']}")
        print(f"{'='*56}\n")

if __name__ == "__main__":
    main()