import csv
import os
import time
from collections import Counter, deque
from statistics import mean

# -----------------------------
# CONFIG
# -----------------------------
SOURCE_MODE = "image"   # "image", "video", or "webcam"
IMAGE_PATH = "test.webp"
VIDEO_PATH = "test.mp4"
OUTPUT_CSV = "quiet_signals_output.txt"
FRAME_SKIP = 5           # analyze every Nth frame for speed
WINDOW_SIZE = 30         # rolling window for volatility
SHOW_WINDOW = True

# -----------------------------
# SCORING FUNCTIONS
# -----------------------------
def compute_frame_burnout_points(emotions):
    """
    emotions example:
    {
        "angry": 2.1,
        "disgust": 0.3,
        "fear": 5.2,
        "happy": 10.4,
        "sad": 20.0,
        "surprise": 1.0,
        "neutral": 61.0
    }
    """
    sad = emotions.get("sad", 0.0)
    angry = emotions.get("angry", 0.0)
    fear = emotions.get("fear", 0.0)
    disgust = emotions.get("disgust", 0.0)
    happy = emotions.get("happy", 0.0)
    neutral = emotions.get("neutral", 0.0)

    # weighted negative load
    negative_score = (
        sad * 1.0 +
        angry * 1.2 +
        fear * 1.1 +
        disgust * 0.8
    ) / 100.0

    points = negative_score

    # flat affect / withdrawal proxy
    if neutral > 85:
        points += 2
    elif neutral > 70:
        points += 1

    # positive affect protection
    if happy > 60:
        points -= 2
    elif happy > 40:
        points -= 1

    return round(points, 2)

def compute_session_risk(avg_points, volatility_ratio, avg_negative, avg_neutral, avg_happy):
    total = avg_points

    # volatility contribution
    if volatility_ratio > 0.50:
        total += 4
    elif volatility_ratio > 0.35:
        total += 2

    # extra persistence logic
    if avg_negative > 45:
        total += 3
    elif avg_negative > 30:
        total += 1.5

    if avg_neutral > 75:
        total += 2

    if avg_happy < 10:
        total += 2

    total = round(total, 2)

    if total >= 20:
        risk = "HIGH"
    elif total >= 10:
        risk = "MODERATE"
    else:
        risk = "LOW"

    return total, risk

def dominant_emotion(emotions):
    return max(emotions, key=emotions.get)

# -----------------------------
# OUTPUT SETUP
# -----------------------------
def init_csv(path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp_sec",
            "dominant_emotion",
            "angry", "disgust", "fear", "happy", "sad", "surprise", "neutral",
            "frame_points"
        ])

def append_csv(path, row):
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)

# -----------------------------
# IMAGE MODE
# -----------------------------
def analyze_image(image_path):
    import cv2  # noqa: F401
    from deepface import DeepFace
    init_csv(OUTPUT_CSV)

    result = DeepFace.analyze(
        img_path=image_path,
        actions=["emotion"],
        enforce_detection=False
    )

    if isinstance(result, list):
        result = result[0]

    emotions = result["emotion"]
    dom = result["dominant_emotion"]
    points = compute_frame_burnout_points(emotions)

    append_csv(OUTPUT_CSV, [
        0,
        dom,
        emotions.get("angry", 0),
        emotions.get("disgust", 0),
        emotions.get("fear", 0),
        emotions.get("happy", 0),
        emotions.get("sad", 0),
        emotions.get("surprise", 0),
        emotions.get("neutral", 0),
        points
    ])

    avg_negative = (
        emotions.get("sad", 0) +
        emotions.get("angry", 0) +
        emotions.get("fear", 0) +
        emotions.get("disgust", 0)
    ) / 4
    avg_neutral = emotions.get("neutral", 0)
    avg_happy = emotions.get("happy", 0)

    total_score, risk = compute_session_risk(
        avg_points=points,
        volatility_ratio=0.0,
        avg_negative=avg_negative,
        avg_neutral=avg_neutral,
        avg_happy=avg_happy
    )

    report = {
        "dominant_emotion": dom,
        "frame_points":     points,
        "avg_negative":     avg_negative,
        "avg_neutral":      avg_neutral,
        "avg_happy":        avg_happy,
        "volatility_ratio": 0.0,
        "total_score":      total_score,
        "risk":             risk,
    }

    print("\n=== QUIET SIGNALS REPORT ===")
    print(f"Dominant emotion: {dom}")
    print(f"Frame burnout points: {points}")
    print(f"Session score: {total_score}")
    print(f"Risk level: {risk}")
    print(f"Saved to: {OUTPUT_CSV}")

    return report

# -----------------------------
# VIDEO / WEBCAM MODE
# -----------------------------
def analyze_stream(capture):
    import cv2
    from deepface import DeepFace
    init_csv(OUTPUT_CSV)

    frame_count = 0
    all_points = []
    dominant_history = []
    rolling_dominants = deque(maxlen=WINDOW_SIZE)
    neg_scores = []
    neutral_scores = []
    happy_scores = []

    start = time.time()

    while True:
        ret, frame = capture.read()
        if not ret:
            break

        frame_count += 1

        if frame_count % FRAME_SKIP != 0:
            if SHOW_WINDOW:
                cv2.imshow("Quiet Signals", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            continue

        try:
            result = DeepFace.analyze(
                img_path=frame,
                actions=["emotion"],
                enforce_detection=False,
                detector_backend="opencv"
            )

            if isinstance(result, list):
                result = result[0]

            emotions = result["emotion"]
            dom = result["dominant_emotion"]
            points = compute_frame_burnout_points(emotions)

            timestamp_sec = round(time.time() - start, 2)

            append_csv(OUTPUT_CSV, [
                timestamp_sec,
                dom,
                emotions.get("angry", 0),
                emotions.get("disgust", 0),
                emotions.get("fear", 0),
                emotions.get("happy", 0),
                emotions.get("sad", 0),
                emotions.get("surprise", 0),
                emotions.get("neutral", 0),
                points
            ])

            all_points.append(points)
            dominant_history.append(dom)
            rolling_dominants.append(dom)

            neg_avg = (
                emotions.get("sad", 0) +
                emotions.get("angry", 0) +
                emotions.get("fear", 0) +
                emotions.get("disgust", 0)
            ) / 4

            neg_scores.append(neg_avg)
            neutral_scores.append(emotions.get("neutral", 0))
            happy_scores.append(emotions.get("happy", 0))

            text = f"{dom} | pts={points:.2f}"

            cv2.putText(
                frame, text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )

        except Exception as e:
            cv2.putText(
                frame, f"Error: {str(e)[:50]}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
            )

        if SHOW_WINDOW:
            cv2.imshow("Quiet Signals", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    capture.release()
    cv2.destroyAllWindows()

    if not all_points:
        print("No usable face/emotion data detected.")
        return

    # volatility = proportion of dominant emotion switches
    switches = 0
    for i in range(1, len(dominant_history)):
        if dominant_history[i] != dominant_history[i - 1]:
            switches += 1

    volatility_ratio = switches / max(1, len(dominant_history) - 1)

    avg_points = mean(all_points)
    avg_negative = mean(neg_scores)
    avg_neutral = mean(neutral_scores)
    avg_happy = mean(happy_scores)

    total_score, risk = compute_session_risk(
        avg_points=avg_points,
        volatility_ratio=volatility_ratio,
        avg_negative=avg_negative,
        avg_neutral=avg_neutral,
        avg_happy=avg_happy
    )

    dom_counts = Counter(dominant_history)

    report = {
        "frames_analyzed":  len(all_points),
        "avg_negative":     avg_negative,
        "avg_neutral":      avg_neutral,
        "avg_happy":        avg_happy,
        "volatility_ratio": volatility_ratio,
        "total_score":      total_score,
        "risk":             risk,
        "dominant_counts":  dict(dom_counts),
    }

    print("\n=== QUIET SIGNALS REPORT ===")
    print(f"Frames analyzed: {len(all_points)}")
    print(f"Average frame points: {avg_points:.2f}")
    print(f"Volatility ratio: {volatility_ratio:.2f}")
    print(f"Average negative emotion load: {avg_negative:.2f}")
    print(f"Average neutral: {avg_neutral:.2f}")
    print(f"Average happy: {avg_happy:.2f}")
    print(f"Session score: {total_score}")
    print(f"Risk level: {risk}")
    print("Dominant emotion counts:", dict(dom_counts))
    print(f"Saved to: {OUTPUT_CSV}")

    return report

# -----------------------------
# QUIET SIGNALS BRIDGE
# -----------------------------
def grayscale_to_signals(report: dict) -> dict:
    """
    Convert a grayscale analysis report into QuietSignals facial signal values.

    report keys (as returned by analyze_image / analyze_stream):
      avg_negative    — average % load from sad + angry + fear + disgust (0–100)
      avg_neutral     — average % neutral expression (0–100)
      avg_happy       — average % happy expression (0–100)
      volatility_ratio — proportion of frames where dominant emotion switched (0–1)

    Returns the three facial signals expected by the QuietSignals composite scorer.
    All values are clipped to [0, 1].
    """
    def clip(v):
        return max(0.0, min(1.0, v))

    # facial_negative_load: normalized negative emotion burden
    # avg_negative of 60% → saturates at 1.0 (very high negative load)
    facial_negative_load = clip(report.get("avg_negative", 0.0) / 60.0)

    # facial_flat_affect: sustained neutral expression as withdrawal proxy
    # avg_neutral of 90% → saturates at 1.0
    facial_flat_affect = clip(report.get("avg_neutral", 0.0) / 90.0)

    # facial_positive_protect: positive affect as protective factor (inverse signal)
    # avg_happy of 60% → saturates at 1.0
    facial_positive_protect = clip(report.get("avg_happy", 0.0) / 60.0)

    return {
        "facial_negative_load":    round(facial_negative_load, 4),
        "facial_flat_affect":      round(facial_flat_affect, 4),
        "facial_positive_protect": round(facial_positive_protect, 4),
    }


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    if SOURCE_MODE == "image":
        analyze_image(IMAGE_PATH)

    elif SOURCE_MODE == "video":
        import cv2
        cap = cv2.VideoCapture(VIDEO_PATH)
        analyze_stream(cap)

    elif SOURCE_MODE == "webcam":
        import cv2
        cap = cv2.VideoCapture(0)
        analyze_stream(cap)

    else:
        print("Invalid SOURCE_MODE. Use image, video, or webcam.")
