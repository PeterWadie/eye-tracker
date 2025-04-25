# main.py
import cv2
import time
import json
import csv
import sys
import platform
from datetime import datetime
from detectors import FaceDetector, LandmarkDetector
from utils import eye_aspect_ratio, get_eye_frame, get_gaze_ratio, get_gaze_direction


def beep():
    """
    Cross-platform system beep:
      - Windows: winsound.Beep at 1 kHz for 200 ms
      - macOS/Linux: BEL character to stdout
    """
    if platform.system() == "Windows":
        import winsound

        winsound.Beep(1000, 200)
    else:
        sys.stdout.write("\a")
        sys.stdout.flush()


def load_config(path="config.json"):
    with open(path, "r") as f:
        return json.load(f)


def main():
    cfg = load_config()

    # --- Video capture setup ---
    cap = cv2.VideoCapture(cfg["camera_index"])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["height"])
    cap.set(cv2.CAP_PROP_FPS, cfg["fps"])

    # --- Detectors ---
    face_det = FaceDetector(
        cfg["dnn_model"]["prototxt"],
        cfg["dnn_model"]["caffemodel"],
        cfg["dnn_model"]["conf_threshold"],
    )
    lm_det = LandmarkDetector(cfg["shape_predictor_path"])

    # --- CSV logging setup ---
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"session_{timestamp_str}.csv"
    log_file = open(log_filename, "w", newline="")
    writer = csv.writer(log_file)
    writer.writerow(
        [
            "timestamp",
            "frame_idx",
            "ear_left",
            "ear_right",
            "avg_ear",
            "gaze_ratio_left",
            "gaze_ratio_right",
            "gaze_state",
            "alarm",
        ]
    )

    blink_start = None
    frame_idx = 0
    prev_alarm = False

    # Color map for gaze text
    gaze_colors = {
        "LEFT": (255, 0, 0),
        "CENTER": (0, 255, 0),
        "RIGHT": (0, 0, 255),
        "UNKNOWN": (255, 255, 255),
    }

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_det.detect(frame)
        alarm = False
        gaze_state = "UNKNOWN"
        gaze_ratio_l = gaze_ratio_r = 0.0

        for rect in faces:
            # Landmarks
            shape = lm_det.detect(gray, rect)
            left_eye_lms = shape[36:42]
            right_eye_lms = shape[42:48]

            # EAR computation
            ear_l = eye_aspect_ratio(left_eye_lms)
            ear_r = eye_aspect_ratio(right_eye_lms)
            avg_ear = (ear_l + ear_r) / 2.0

            # Blink / fatigue detection
            if avg_ear < cfg["ear_blink_threshold"]:
                if blink_start is None:
                    blink_start = time.time()
                elif (time.time() - blink_start) > cfg["ear_closed_secs"]:
                    alarm = True
            else:
                blink_start = None

            # Gaze estimation
            gaze_state = get_gaze_direction(shape, frame)
            # Also get raw ratios for logging
            eye_frame_l = get_eye_frame(frame, left_eye_lms)
            eye_frame_r = get_eye_frame(frame, right_eye_lms)
            gaze_ratio_l = get_gaze_ratio(eye_frame_l)
            gaze_ratio_r = get_gaze_ratio(eye_frame_r)

            # Visualization: eye contours + EAR
            for eye_lms in (left_eye_lms, right_eye_lms):
                cv2.polylines(frame, [eye_lms], True, (0, 255, 0), 1)
            cv2.putText(
                frame,
                f"EAR: {avg_ear:.2f}",
                (rect[0], rect[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

            # Alarm overlay
            if alarm:
                cv2.putText(
                    frame,
                    "ALARM!",
                    (rect[0], rect[1] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

        # Beep once on new alarm event
        if alarm and not prev_alarm:
            beep()
        prev_alarm = alarm

        # Show colored gaze text in top-left
        color = gaze_colors.get(gaze_state, gaze_colors["UNKNOWN"])
        cv2.putText(
            frame,
            f"Gaze: {gaze_state}",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

        # Logging (down-sample every Nth frame)
        if frame_idx % cfg["log_downsample_rate"] == 0:
            writer.writerow(
                [
                    datetime.now().isoformat(),
                    frame_idx,
                    f"{ear_l:.3f}",
                    f"{ear_r:.3f}",
                    f"{avg_ear:.3f}",
                    f"{gaze_ratio_l:.3f}",
                    f"{gaze_ratio_r:.3f}",
                    gaze_state,
                    int(alarm),
                ]
            )

        cv2.imshow("Eye-Tracker", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()
    log_file.close()


if __name__ == "__main__":
    main()
