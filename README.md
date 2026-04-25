# Serve Contact Detection

Detects the frame where racket–ball contact occurs in a serve video.

**Accuracy:** ±1–2 frames for 4/5 videos  
Video 5 is handled with lower confidence due to distance and multiple players.

---

## Core Idea

Instead of detecting the ball (which is small, fast, and often blurred),  
I track the **hitting wrist using MediaPipe Pose**.

At contact:
- The arm is fully extended  
- The wrist reaches its **highest point** (min Y)  
- In some cases, **velocity peak** is more accurate than height  

So I use both:
- **Height peak**
- **Velocity peak**

and select the better signal per video.

---

## Pipeline

1. Extract pose (wrist, elbow, shoulder)
2. Auto-detect hitting arm  
3. Track wrist trajectory (smoothed)
4. Detect swing phase
5. Compute height & velocity peaks  
6. Select contact frame

---

## Output

- **JSON (saved in the `output/` folder):**
  - `contact_frame`
  - `timestamp`
  - `hitting_arm`
  - `signal_agreement` (confidence)

- **Annotated video ([View Annotated Videos on Google Drive](https://drive.google.com/drive/folders/11HtKGffC6B_GmOk_LbtfM1K2uTUJvmr7?usp=drive_link)):**
  - Wrist tracking
  - Contact frame in green
  - Swing phases

---

## How to Run

```bash
git clone https://github.com/dhatrisriram/Drop-it.git
cd Drop-it
pip install -r requirements.txt
python download_videos.py
```
To run on a single video:

```bash
python detect_contact.py --video videos/serve_01.mp4
```
Run on all videos:

```bash
python detect_contact.py --all
```
Generate annotated video:

```bash
python detect_contact.py --video videos/serve_01.mp4 --annotate
```
## Key Challenges

- **No single rule works across all serves:** Pose noise and variation in motion styles mean different signals are needed for different videos.
- **Video 5:** Players are far away, the racket is extremely small, and multiple people in the frame make it hard to reliably identify the server.

## Design Choices

- **MediaPipe Pose:** Fast, CPU-friendly, and provides built-in visibility scores.
- **Wrist-based detection:** More reliable than trying to track a blurred 15px ball.
- **Combined signals:** Using multiple temporal signals (height and velocity) instead of relying on a single, fragile rule.

## Limitations

- Video 5 is less accurate due to weak visual cues and multi-person ambiguity.

## Potential Improvements (Given More Time)

- **Lightweight CNN on wrist trajectory:** Classify whether the best contact signal is the height or velocity peak, removing any per-video manual adjustment.
- **Player selection by bounding box size:** For video 5, use the person with the largest bounding box (closest to camera) as the server.
- **TrackNet fine-tuning:** Add a ball-tracking model to further boost confidence when the ball is visible.
- **Optical flow:** Use dense flow to track the racket head directly when landmarks are noisy.
