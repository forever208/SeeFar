"""
key component of DeepSORT algorithm:
    1. matching cascade
    2. distance measurements
    3. Kalman filter to predict tracks and update tracks
"""

from __future__ import absolute_import
import numpy as np
from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from .track import Track


class Tracker:
    """
    Manage all Track instances

    Attributes
    ----------
        metric : nn_matching.NearestNeighborDistanceMetric
            The distance metric used for measurement to track association.
        kf : kalman_filter.KalmanFilter
            A Kalman filter to filter target trajectories in image space.
        tracks : List[Track]
            The list of active tracks at the current time step.
    """

    def __init__(self, metric, max_iou_distance=0.7, max_age=70, n_init=3):
        """
        @param metric: class instance of distance metric
        @param max_iou_distance:
        @param max_age: max number of missing frames when an unmatched track is deleted
        @param n_init: number of frames when an unmatched detection is initialised as a new track
        """

        self.metric = metric    # distance metric instance (cosine/euclidean)
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init

        self.kf = kalman_filter.KalmanFilter()    # instantiate a Kalman filter
        self.tracks = []    # list of active tracks at the current time step
        self._next_id = 1    # ID counter


    def predict(self):
        """
        predict the active tracks for next frame by Kalman filter
        This function should be called before 'update()' for every time step
        """
        for track in self.tracks:
            track.predict(self.kf)


    def update(self, detections):
        """
        Perform bbox matching and track management.
        @param detections: detected bbox list [detection, detection...],
                           each detection is a class instance with attributes (det.confidence, det.feature, det.tlwh)
        """

        """1. Matching cascade + IOU matching"""
        matches, unmatched_tracks, unmatched_detections = self._match(detections)

        """2. Track and detection management"""
        #    1) for matched detection-track pair, use track and detection to generate better track by Kalman filter
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(self.kf, detections[detection_idx])
        
        #    2) for unmatched tracks, mark as missed
        #       (delete the confirmed track if no matched detection in 70 consecutive frames)
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        
        #    3) for unmatched detections， initialise as new tracks
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])

        """3. update track list (only keep confirmed and tentative tracks)"""
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        """4. get the ID and features of confirmed tracks"""
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
            track.features = []

        """5. Update the feature history for matched (track, detection) pairs """
        self.metric.partial_fit(np.asarray(features), np.asarray(targets), active_targets)


    def _match(self, detections):
        """
        matching cascade + IOU matching
        @param detections: detected bbox list [detection, detection...]
        @return:
        """

        def gated_metric(tracks, dets, track_indices, detection_indices):
            """
            return the cost matrix based on features distance + Mahala distance
            """
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            # compute the cost matrix by the appearance feature distance between detections and tracks
            cost_matrix = self.metric.distance(features, targets)
            # update the cost matrix by computing the Mahala distance between detection bbox and track bbox
            cost_matrix = linear_assignment.gate_cost_matrix(self.kf, cost_matrix, tracks, dets, track_indices, detection_indices)
            return cost_matrix

        # divide active tracks of current frame into confirmed and tentative tracks
        confirmed_tracks = [i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # 1. perform matching cascade between confirmed tracks and newly detections
        #    input cost matrix (features distance + Mahala distance)
        #    return matches, unmatched tracks, unmatched detections
        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(gated_metric, self.metric.matching_threshold, self.max_age, self.tracks, detections, confirmed_tracks)

        # 2. perform IOU matching between (tentative + unmatched) tracks and unmatched detections
        #    iou_track_candidates = tentative tracks + unmatched tracks
        #    input cost matrix (IOU metrics)
        iou_track_candidates = unconfirmed_tracks + [k for k in unmatched_tracks_a if self.tracks[k].time_since_update == 1]
        unmatched_tracks_a = [k for k in unmatched_tracks_a if self.tracks[k].time_since_update != 1]    # 并非刚刚没有匹配上的轨迹
        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(iou_matching.iou_cost, self.max_iou_distance, self.tracks, detections, iou_track_candidates, unmatched_detections)

        # 3. combine the above 2 matches
        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections


    def _initiate_track(self, detection):
        """
        mark unmatched detections as new tentative track, it has 3 frames tentative period
        @param detection: detected bbox list [detection, detection...],
                          each detection is a class instance with attributes (det.confidence, det.feature, det.tlwh)
        """
        mean, covariance = self.kf.initiate(detection.to_xyah())
        self.tracks.append(Track(mean, covariance, self._next_id, self.n_init, self.max_age, detection.feature))
        self._next_id += 1
