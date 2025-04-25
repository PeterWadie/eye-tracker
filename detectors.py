# detectors.py
import cv2
import dlib
import numpy as np
from utils import shape_to_np


class FaceDetector:
    def __init__(self, proto_path, model_path, conf_thresh=0.5):
        self.net = cv2.dnn.readNetFromCaffe(proto_path, model_path)
        self.conf_thresh = conf_thresh

    def detect(self, frame):
        """Return list of (startX, startY, endX, endY) for faces."""
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
        )
        self.net.setInput(blob)
        detections = self.net.forward()
        rects = []
        for i in range(0, detections.shape[2]):
            conf = detections[0, 0, i, 2]
            if conf > self.conf_thresh:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                rects.append((startX, startY, endX, endY))
        return rects


class LandmarkDetector:
    def __init__(self, predictor_path):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)

    def detect(self, gray, rect):
        """Given a gray frame and a face rect, return 68 landmarks as NumPy array."""
        # wrap rect into dlib.rectangle if needed
        if not isinstance(rect, dlib.rectangle):
            rect = dlib.rectangle(rect[0], rect[1], rect[2], rect[3])
        shape = self.predictor(gray, rect)
        return shape_to_np(shape)
