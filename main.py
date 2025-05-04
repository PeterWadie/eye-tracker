# main.py

import os
import sys
import platform
import cv2
import time
import json
import csv
import numpy as np
from collections import deque
from datetime import datetime
from detectors import FaceDetector, LandmarkDetector
from utils import eye_aspect_ratio, get_eye_frame, get_gaze_ratio, get_gaze_direction


def beep():
    """
    Cross-platform system beep:
      - Windows: winsound.Beep at 1 kHz for 200 ms
      - macOS: afplay the system 'Ping' sound
      - Linux/others: BEL character
    """
    if platform.system() == "Windows":
        import winsound

        winsound.Beep(1000, 200)
    elif platform.system() == "Darwin":
        os.system("afplay /System/Library/Sounds/Ping.aiff")
    else:
        sys.stdout.write("\a")
        sys.stdout.flush()


def load_config(path="config.json"):
    with open(path, "r") as f:
        return json.load(f)


def save_config(cfg, path="config.json"):
    with open(path, "w") as f:
        json.dump(cfg, f, indent=4)


def main():
    cfg = load_config()

    # --- Video & detectors setup ---
    cap = cv2.VideoCapture(cfg["camera_index"])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["height"])
    cap.set(cv2.CAP_PROP_FPS, cfg["fps"])

    face_det = FaceDetector(
        cfg["dnn_model"]["prototxt"],
        cfg["dnn_model"]["caffemodel"],
        cfg["dnn_model"]["conf_threshold"],
    )
    lm_det = LandmarkDetector(cfg["shape_predictor_path"])

    # --- Calibration (unchanged) ---
    # ... assume you kept your calibrate_ear() from earlier ...

    # Uncomment to re-calibrate on each run:
    # new_thresh = calibrate_ear(cap, face_det, lm_det, cfg, 5, 5)
    # cfg["ear_blink_threshold"] = new_thresh
    # save_config(cfg)

    # --- Prepare smoothing & fatigue counters ---
    smooth_window = int(cfg["fps"] * 0.5) or 1  # 0.5s window
    ear_window = deque(maxlen=smooth_window)
    closed_frames = 0
    closed_thresh = int(cfg["fps"] * cfg["ear_closed_secs"])

    # --- CSV logging setup ---
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_fname = f"session_{ts}.csv"
    f = open(log_fname, "w", newline="")
    writer = csv.writer(f)
    writer.writerow(
        [
            "timestamp",
            "frame_idx",
            "ear_left",
            "ear_right",
            "avg_ear",
            "smoothed_ear",
            "gaze_ratio_left",
            "gaze_ratio_right",
            "gaze_state",
            "alarm",
        ]
    )

    prev_alarm = False
    frame_idx = 0

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
        gaze_l = gaze_r = 0.0

        for rect in faces:
            shape = lm_det.detect(gray, rect)
            le_lms, re_lms = shape[36:42], shape[42:48]

            # raw EAR
            ear_l = eye_aspect_ratio(le_lms)
            ear_r = eye_aspect_ratio(re_lms)
            avg_ear = (ear_l + ear_r) / 2.0

            # smoothing
            ear_window.append(avg_ear)
            smooth_ear = float(np.median(ear_window))

            # fatigue via frame counts
            if smooth_ear < cfg["ear_blink_threshold"]:
                closed_frames += 1
                if closed_frames >= closed_thresh:
                    alarm = True
            else:
                closed_frames = 0

            # gaze
            gaze_state = get_eye_frame  # placeholder to avoid lint error
            gaze_state = get_gaze_direction(shape, frame)
            gaze_l = get_gaze_ratio(get_eye_frame(frame, le_lms))
            gaze_r = get_gaze_ratio(get_eye_frame(frame, re_lms))

            # draw eyes & EAR
            for eye_lms in (le_lms, re_lms):
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

        # beep once per alarm onset
        if alarm and not prev_alarm:
            beep()
        prev_alarm = alarm

        # show gaze
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

        # log (down-sample)
        if frame_idx % cfg["log_downsample_rate"] == 0:
            writer.writerow(
                [
                    datetime.now().isoformat(),
                    frame_idx,
                    f"{ear_l:.3f}",
                    f"{ear_r:.3f}",
                    f"{avg_ear:.3f}",
                    f"{smooth_ear:.3f}",
                    f"{gaze_l:.3f}",
                    f"{gaze_r:.3f}",
                    gaze_state,
                    int(alarm),
                ]
            )

        cv2.imshow("Eye-Tracker", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()
    f.close()


if __name__ == "__main__":
    main()
