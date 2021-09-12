# vim: expandtab:ts=4:sw=4
import numpy as np


def _pdist(a, b):
    """
    Compute pair-wise squared distance between vectors
    @param a: An NxM matrix of N samples of dimensionality M.
    @param b: An LxM matrix of L samples of dimensionality M.
    @return: 2D matrix (N, M),, element (i, j) contains the squared distance between `a[i]` and `b[j]`.

    参考：https://blog.csdn.net/frankzd/article/details/80251042
    """

    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a2, b2 = np.square(a).sum(axis=1), np.square(b).sum(axis=1)
    r2 = -2. * np.dot(a, b.T) + a2[:, None] + b2[None, :]
    r2 = np.clip(r2, 0., float(np.inf))
    return r2


def _cosine_distance(a, b, data_is_normalized=False):
    """
    Compute pair-wise cosine distance between points in `a` and `b`.
    @param a: An NxM matrix of N samples of dimensionality M.
    @param b: An LxM matrix of L samples of dimensionality M.
    @param data_is_normalized: Optional[bool], If True, assumes rows in a and b are unit length vectors.
                               Otherwise, a and b are explicitly normalized to lenght 1.
    @return: 2D matrix (N, M), element (i, j) contains the squared distance between `a[i]` and `b[j]`.

    参考: https://blog.csdn.net/u013749540/article/details/51813922
    """

    if not data_is_normalized:
        a = np.asarray(a) / np.linalg.norm(a, axis=1, keepdims=True)    # np.linalg.norm 求向量的范数，默认是 L2 范数
        b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)
    return 1. - np.dot(a, b.T)    # 余弦距离 = 1 - 余弦相似度


def _nn_euclidean_distance(x, y):
    """
    Helper function for nearest neighbor distance metric (Euclidean).
    @param x: A matrix of N row-vectors (sample points).
    @param y: A matrix of M row-vectors (query points).
    @return: A vector of length M that contains the smallest Euclidean distance between x and y
    """
    distances = _pdist(x, y)
    return np.maximum(0.0, distances.min(axis=0))


def _nn_cosine_distance(x, y):
    """
    Helper function for nearest neighbor distance metric (cosine).
    @param x: A matrix of N row-vectors (sample points).
    @param y: A matrix of M row-vectors (query points).
    @return: A vector (1, M) that contains the smallest cosine distance between x and y
    """
    distances = _cosine_distance(x, y)
    return distances.min(axis=0)


class NearestNeighborDistanceMetric(object):
    """
    A nearest neighbor distance metric,
    for each track, returns the closest detections
    """

    def __init__(self, metric, matching_threshold, budget=None):
        """
        @param metric: "euclidean" or "cosine"
        @param matching_threshold: samples with larger distance are considered an invalid match
        @param budget: fix samples per class to at most this number. Removes history samples when the budget is reached.
        """
        if metric == "euclidean":
            self._metric = _nn_euclidean_distance
        elif metric == "cosine":
            self._metric = _nn_cosine_distance
        else:
            raise ValueError("Invalid metric; must be either 'euclidean' or 'cosine'")
        self.matching_threshold = matching_threshold
        self.budget = budget    # default as 100, 用于控制 feature 的数目

        # A dictionary that maps from track ID to detections
        self.samples = {}


    def partial_fit(self, features, targets, active_targets):
        """
        Update the matched track-detection pairs before computing the distance
        @param features: 2D array (N, 512), N detections with 512D features
        @param targets: 1D array (N), ID of confirmed tracks
        @param active_targets: int list of confirmed (matched) tracks ID
        """

        for feature, target in zip(features, targets):
            # samples is a dict {target: feature list}
            self.samples.setdefault(target, []).append(feature)

            # only maintain 100 feature history for each matched detections
            if self.budget is not None:
                self.samples[target] = self.samples[target][-self.budget:]
        
        # only keep the active target
        self.samples = {k: self.samples[k] for k in active_targets}


    def distance(self, features, targets):
        """
        Compute distance between detections and tracks.
        @param features: 2D array (N, 512), N detections with 512D features.
        @param targets: int list of track IDs
        @return:
            cost_matrix: 2D cost matrix (num_targets, num_detections)
                         element (i, j) contains the closest squared distance between `targets[i]` and `features[j]`.
        """

        cost_matrix = np.zeros((len(targets), len(features)))
        for i, target in enumerate(targets):
            cost_matrix[i, :] = self._metric(self.samples[target], features)

        return cost_matrix
