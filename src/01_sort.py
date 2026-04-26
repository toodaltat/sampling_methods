import numpy as np


def iou(box_a, box_b):
    """
    Compute Intersection over Union between two boxes.
    Boxes are [x1, y1, x2, y2]
    """
    xA = max(box_a[0], box_b[0])
    yA = max(box_a[1], box_b[1])
    xB = min(box_a[2], box_b[2])
    yB = min(box_a[3], box_b[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])

    union = area_a + area_b - inter_area
    if union == 0:
        return 0.0

    return inter_area / union


class Track:
    def __init__(self, box, track_id):
        self.box = np.array(box[:4], dtype=float)
        self.id = track_id
        self.missed = 0

    def update(self, box):
        self.box = np.array(box[:4], dtype=float)
        self.missed = 0


class Sort:
    """
    A simple SORT-like tracker.

    Input to update():
        numpy array of shape (N, 5)
        each row = [x1, y1, x2, y2, score]

    Output from update():
        numpy array of shape (M, 5)
        each row = [x1, y1, x2, y2, track_id]
    """

    def __init__(self, max_age=10, iou_threshold=0.3):
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self.tracks = []
        self.next_id = 1

    def update(self, detections):
        """
        detections: np.array of shape (N, 5)
        returns: np.array of shape (M, 5)
        """
        if detections is None or len(detections) == 0:
            # No detections this frame: age all tracks
            survivors = []
            for track in self.tracks:
                track.missed += 1
                if track.missed <= self.max_age:
                    survivors.append(track)
            self.tracks = survivors

            return np.array(
                [[t.box[0], t.box[1], t.box[2], t.box[3], t.id] for t in self.tracks],
                dtype=float
            ) if self.tracks else np.empty((0, 5))

        detections = np.asarray(detections, dtype=float)

        matched_tracks = set()
        matched_dets = set()

        # Greedy IoU matching
        pairs = []
        for t_idx, track in enumerate(self.tracks):
            for d_idx, det in enumerate(detections):
                score = iou(track.box, det[:4])
                if score >= self.iou_threshold:
                    pairs.append((score, t_idx, d_idx))

        pairs.sort(reverse=True, key=lambda x: x[0])

        for score, t_idx, d_idx in pairs:
            if t_idx in matched_tracks or d_idx in matched_dets:
                continue
            self.tracks[t_idx].update(detections[d_idx])
            matched_tracks.add(t_idx)
            matched_dets.add(d_idx)

        # Age unmatched tracks
        survivors = []
        for t_idx, track in enumerate(self.tracks):
            if t_idx not in matched_tracks:
                track.missed += 1
            if track.missed <= self.max_age:
                survivors.append(track)
        self.tracks = survivors

        # Add new tracks for unmatched detections
        for d_idx, det in enumerate(detections):
            if d_idx not in matched_dets:
                self.tracks.append(Track(det, self.next_id))
                self.next_id += 1

        # Return active tracks
        results = []
        for track in self.tracks:
            x1, y1, x2, y2 = track.box
            results.append([x1, y1, x2, y2, track.id])

        return np.array(results, dtype=float) if results else np.empty((0, 5))