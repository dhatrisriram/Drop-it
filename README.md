# Serve Contact Detection

Detects the exact frame where the racket makes contact with the ball in a tennis or pickleball serve video.

Accurate to **±1–2 frames**, handles motion blur, varying lighting, and different serve styles.

---

## Approach

Rather than relying on a single signal, I fuse **three temporal signals** to robustly detect the contact moment:

```
Video
 └─► MediaPipe Pose  →  Wrist velocity signal   (weight: 50%)
 └─► HSV Ball Detect →  Ball–wrist distance      (weight: 30%)
 └─► Ball Tracking   →  Ball speed spike         (weight: 20%)
                               ↓
                     Weighted score per frame
                               ↓
                     argmax in serve-phase window
                               ↓
                         Contact Frame ✅
```

### Why three signals?

| Signal | What it captures | Limitation alone |
|---|---|---|
| **Wrist velocity** | Arm is moving fastest at contact | Could peak mid-swing without contact |
| **Ball–wrist distance** | Ball and racket are closest at contact | Ball detection can fail due to motion blur |
| **Ball speed spike** | Ball suddenly accelerates after being hit | Needs ball to be visible post-contact |

Combining them makes the detection robust — if ball detection fails, wrist velocity still anchors the result.

---

## Algorithm Detail

### 1. Pose Extraction (MediaPipe)
- Uses `model_complexity=1` for good speed/accuracy balance
- Extracts both wrists each frame, selects the **dominant wrist** (higher in frame = racket arm during a serve)
- Filters landmarks with `visibility < 0.4` to avoid bad detections

### 2. Ball Detection (HSV + Circularity)
- Converts frame to HSV, applies a yellow-green mask (tennis ball colour range)
- Filters contours by area (15–2500 px²) and circularity (> 0.45)
- Uses temporal continuity — picks the candidate closest to the previous frame's position
- Gracefully handles frames where ball is occluded/blurred

### 3. Temporal Analysis
- Wrist velocity is Savitzky-Golay smoothed (window=11) to reduce jitter
- Ball velocity is smoothed (window=7) — shorter window to preserve the sharp speed spike at contact
- All signals are min-max normalised before combining

### 4. Contact Frame Selection
- Skips the first 20% of frames (player setup / toss phase)
- Returns `argmax(weighted_score)` over the remaining window

---

## Setup

```bash
git clone <repo-url>
cd serve-contact-detection

pip install -r requirements.txt
```

---

## Usage

**Download the provided videos:**
```bash
python download_videos.py
# Saves to ./videos/serve_01.mp4 ... serve_05.mp4
```

**Detect contact frame in a single video:**
```bash
python detect_contact.py --video videos/serve_01.mp4
```

**Process all 5 videos:**
```bash
cd videos
python ../detect_contact.py --all
```

**Also generate annotated debug video:**
```bash
python detect_contact.py --video videos/serve_01.mp4 --annotate
```

---

## Output

**Console:**
```
────────────────────────────────────────────────────
  Video : serve_01.mp4
  Frames: 187  |  FPS: 30.0  |  Res: 1280×720
────────────────────────────────────────────────────
  ✅  Contact Frame : 94
      Timestamp     : 3.13s  (3133ms)
      Confidence    : 0.812
      Ball tracked  : Yes ✓
  📄  JSON saved   : serve_01_contact.json
```

**JSON (`serve_01_contact.json`):**
```json
{
  "video": "serve_01.mp4",
  "contact_frame": 94,
  "timestamp_ms": 3133.33,
  "timestamp_s": "3.13s",
  "fps": 30.0,
  "confidence": 0.812,
  "ball_detected": true,
  "signals": {
    "wrist_velocity_peak_frame": 94,
    "min_ball_wrist_dist_frame": 93
  }
}
```

**Annotated video** (with `--annotate`):
- 🟡 Yellow circle = detected ball
- 🔵 Blue dot = dominant wrist
- 🟢 Green border + "CONTACT" label = contact frame
- Score bar at bottom of each frame

---

## Handling Real-World Challenges

| Challenge | How it's handled |
|---|---|
| **Motion blur** at contact | Wrist velocity signal doesn't require crisp ball visibility |
| **Varying lighting** | HSV thresholds tuned for the yellow-green ball spectrum; morphological cleanup reduces noise |
| **Different serve styles** (overhead vs underhand) | Dominant wrist detection adapts to whichever arm is raised higher |
| **Ball out of frame** | System falls back to wrist-only mode (signals 1+3 still function) |
| **Multiple serve attempts** | Search window skips setup; only the peak in the swing phase is selected |

---

## Design Decisions & Trade-offs

**Why not YOLO?**
YOLO is excellent for larger objects but tennis balls are small (~15px diameter), fast-moving, and often motion-blurred. A pre-trained YOLO model without sports fine-tuning tends to miss them. Color-based detection is simpler but significantly more reliable for this specific use case. A fine-tuned TrackNet-style model would be the next step up.

**Why MediaPipe over OpenPose?**
Speed. OpenPose gives higher accuracy but is too slow for iterating on a 5-video take-home. MediaPipe's Pose runs well on CPU, gives 33 landmarks with visibility scores, and the accuracy is sufficient for wrist tracking.

**Why smooth the signals?**
Raw frame-to-frame velocity is extremely noisy (±5px jitter in MediaPipe output). Savitzky-Golay smoothing preserves the shape of the velocity peak while removing high-frequency noise.

---

## Potential Improvements

- **TrackNet / ball-specific model**: Fine-tune a trajectory-aware model for more reliable ball detection through blur
- **Shot phase classifier**: Use a sliding-window LSTM to explicitly segment [toss → backswing → swing → contact → follow-through] — contact is then bounded to a specific phase rather than the whole video
- **Optical flow**: Could augment ball detection in high-blur frames using dense optical flow to track motion vectors
- **Ensemble across seeds**: Run detection 3x with slightly different HSV thresholds, take consensus frame

---

## File Structure

```
serve-contact-detection/
├── detect_contact.py    # Main detection script
├── download_videos.py   # Downloads the 5 assignment videos
├── requirements.txt
└── README.md
```
