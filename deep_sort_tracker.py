"""
deep_sort_tracker.py
---------------------
Wrapper class for DeepSORT tracker.

Handles:
- Loading the appearance feature encoder (MARS model)
- Converting YOLO detections for DeepSORT
- Running prediction + update steps
- Storing confirmed tracks

Author: Jeshwanth Ganesh
"""
from deep_sort.deep_sort.tracker import Tracker as DeepSortTracker
from deep_sort.tools import generate_detections as gdet
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
import numpy as np


class Tracker:
    """
    Tracker class that wraps DeepSORT functionality.
    """

    def __init__(self):
        # Metric for similarity matching
        max_cosine_distance = 0.4
        nn_budget = None

        # Path to MARS appearance model
        encoder_model_filename = (
            r'model_data\mars-small128.pb'
        )

        # Initialize metric + DeepSORT tracker
        metric = nn_matching.NearestNeighborDistanceMetric(
            "cosine", max_cosine_distance, nn_budget
        )
        self.tracker = DeepSortTracker(metric)

        # Load appearance feature encoder
        self.encoder = gdet.create_box_encoder(encoder_model_filename, batch_size=1)

        self.tracks = []  # Store final tracks (clean format)

    # --------------------------------------------------
    def update(self, frame, detections):
        """
        Updates tracker with new detection results.

        Args:
            frame: Current video frame (BGR)
            detections: [[x1,y1,x2,y2,score], ...]
        """

        # When no detections → only predict
        if len(detections) == 0:
            self.tracker.predict()
            self.tracker.update([])
            self.update_tracks()
            return

        # Convert bbox format → (x, y, w, h)
        bboxes = np.asarray([d[:-1] for d in detections])
        bboxes[:, 2:] = bboxes[:, 2:] - bboxes[:, 0:2]
        scores = [d[-1] for d in detections]

        # Generate appearance features
        features = self.encoder(frame, bboxes)

        # Create DeepSORT Detection objects
        dets = [
            Detection(bbox, scores[i], features[i])
            for i, bbox in enumerate(bboxes)
        ]

        # Step 1: predict positions
        self.tracker.predict()
        # Step 2: update with detections
        self.tracker.update(dets)

        # Store confirmed tracks
        self.update_tracks()

    # --------------------------------------------------
    def update_tracks(self):
        """
        Converts DeepSORT internal tracks → simple Track objects.
        """
        tracks = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            bbox = track.to_tlbr()  # top-left bottom-right format
            tracks.append(Track(track.track_id, bbox))

        self.tracks = tracks


class Track:
    """
    Simple container for an active track.
    """

    def __init__(self, id, bbox):
        self.track_id = id
        self.bbox = bbox
