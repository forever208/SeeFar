# vim: expandtab:ts=4:sw=4


class TrackState:
    """
    Newly created tracks are classified as `tentative`
    Once before enough evidence has been observed. the track state is changed to `confirmed`.
    Tracks that are no longer alive are classified as `deleted` and removed from active tracks.
    """

    Tentative = 1
    Confirmed = 2
    Deleted = 3


class Track:
    """
    A single track with state space [x, y, a, h] and associated velocities,
    where `[x, y]` is the center of the bounding box, `a` is the aspect ratio and `h` is the height.
    """

    def __init__(self, mean, covariance, track_id, n_init, max_age, feature=None):
        """
        @param mean: 1D array (8, ), mean vector of the initial state distribution.
        @param covariance: 2D array (8, 8), covariance matrix of the initial state distribution.
        @param track_id: int, a unique track ID.
        @param n_init: default as 3, number of consecutive detections before the track is confirmed.
                       The track state is set to `Deleted` once a miss occurs within `n_init` frames.
        @param max_age: default as 70, The maximum number of consecutive misses a confirmed track can have
        @param feature: 1D array (512, ), track feature which is added to the self.features (feature history)
        """

        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id
        self.hits = 1    # number of matches，+1 when self.update is called, if hits>=n_init，the track ==> Confirmed
        self.age = 1    # duplicate attributes with self.time_since_update
        self.time_since_update = 0    # +1 when self.predict() is called. set as 0 when self.update is called
        self.state = TrackState.Tentative    # initialise a new track as Tentative (because it needs 3 hits)
        self.features = []    # feature history of a track, add new feature if self.update() is called
        if feature is not None:
            self.features.append(feature)
        self._n_init = n_init    # 3: tentative period of 3 frames to transform a tentative track into confirmed track
        self._max_age = max_age    # 70: delete the confirmed track if no matched detection in 70 consecutive frames


    def to_tlwh(self):
        """
        Get current position in bounding box format `(top left x, top left y, width, height)`
        """
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret


    def to_tlbr(self):
        """
        Get current position in bounding box format `(min x, miny, max x, max y)`.
        """
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret


    def predict(self, kf):
        """
        Perform Kalman filter prediction step to propagate the state distribution one time step further
        @param kf: class instance, kalman_filter.KalmanFilter
        """
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1


    def update(self, kf, detection):
        """
        Perform Kalman filter measurement update step and update the feature cache.
        This method is called after a successful match
        @param kf: class instance, kalman_filter.KalmanFilter
        @param detection: The associated detection.
        """
        self.mean, self.covariance = kf.update(self.mean, self.covariance, detection.to_xyah())
        self.features.append(detection.feature)
        self.hits += 1
        self.time_since_update = 0    # reset to 0 once a track has a successful match

        # if a tentative track is matched for 3 consecutive frames, the tentative track is confirmed as a track.
        # otherwise, the tentative track will be deleted in self.mark_missed()
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed


    def mark_missed(self):
        """
        Mark the track as missed if it has no successful match during matching cascade and IOU matching
        """

        # delete the tentative track if no matched detections after 3 frames (missing match is tolerated in 3 frames)
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        # delete the confirmed track if no matched detection in 70 consecutive frames
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted


    def is_tentative(self):
        """
        Returns True if this track is tentative (unconfirmed).
        """
        return self.state == TrackState.Tentative


    def is_confirmed(self):
        """
        Returns True if this track is confirmed.
        """
        return self.state == TrackState.Confirmed


    def is_deleted(self):
        """
        Returns True if this track is dead and should be deleted.
        """
        return self.state == TrackState.Deleted
