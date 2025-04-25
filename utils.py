# utils.py
import numpy as np
import dlib
import cv2


def shape_to_np(shape: dlib.full_object_detection) -> np.ndarray:
    """
    Convert Dlib shape (68 points) to a NumPy array of shape (68, 2).
    """
    coords = np.zeros((68, 2), dtype="int")
    for i in range(68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


def eye_aspect_ratio(eye: np.ndarray) -> float:
    """
    Compute the eye aspect ratio (EAR) for a given eye landmarks array of shape (6,2).
    """
    # vertical distances
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    # horizontal distance
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)


def get_eye_frame(frame: np.ndarray, eye_landmarks: np.ndarray) -> np.ndarray:
    """
    Extract a tight-cropped BGR image of the eye region given 6 (x,y) landmarks.
    """
    # Create mask and isolate eye
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [eye_landmarks], 255)
    eye = cv2.bitwise_and(frame, frame, mask=mask)

    # Crop to bounding box + small margin
    x1, y1 = np.min(eye_landmarks, axis=0)
    x2, y2 = np.max(eye_landmarks, axis=0)
    margin = 2
    return eye[y1 - margin : y2 + margin, x1 - margin : x2 + margin]


def get_gaze_ratio(eye_frame: np.ndarray, thresh_val: int = 70) -> float:
    """
    Threshold the eye crop and compute the ratio of white pixels
    in left vs. right halves. Returns (left_count / (right_count + 1)).
    """
    gray = cv2.cvtColor(eye_frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)
    h, w = thresh.shape
    left_side = thresh[:, : w // 2]
    right_side = thresh[:, w // 2 :]
    left_count = cv2.countNonZero(left_side)
    right_count = cv2.countNonZero(right_side)
    return left_count / (right_count + 1)


def get_gaze_direction(
    shape: np.ndarray,
    frame: np.ndarray,
    thresh_val: int = 70,
    left_ratio_thresh: float = 1.2,
    right_ratio_thresh: float = 0.8,
) -> str:
    """
    Extract both eyes, compute their gaze ratios, average them,
    and classify into "LEFT", "CENTER", or "RIGHT".
    """
    # Landmark indices for left/right eyes
    left_eye_lms = shape[36:42]
    right_eye_lms = shape[42:48]

    # Crop eye regions
    left_eye = get_eye_frame(frame, left_eye_lms)
    right_eye = get_eye_frame(frame, right_eye_lms)

    # Compute ratios
    ratio_l = get_gaze_ratio(left_eye, thresh_val)
    ratio_r = get_gaze_ratio(right_eye, thresh_val)
    avg_ratio = (ratio_l + ratio_r) / 2.0

    if avg_ratio > left_ratio_thresh:
        return "LEFT"
    elif avg_ratio < right_ratio_thresh:
        return "RIGHT"
    else:
        return "CENTER"
