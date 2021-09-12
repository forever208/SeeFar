# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
# The linear sum assignment problem is also known as minimum weight matching in bipartite graphs.
from scipy.optimize import linear_sum_assignment as linear_assignment
from . import kalman_filter

INFTY_COST = 1e+5


def min_cost_matching(distance_metric, max_distance, tracks, detections, track_indices=None, detection_indices=None):
    """
    Hungarian assignment algorithm
    cost matrix can be IOU / cosine feature and Mahala distance

    Parameters
    ----------
    distance_metric : Callable[List[Track], List[Detection], List[int], List[int]) -> ndarray
        Given a list of tracks and detections, a list of N track indices and M detection indices.
        The metric should return the 2D cost matrix (N, M),
        where element (i, j) is the cost between the i-th track and the j-th detection.
    max_distance : float
        Gating threshold. Associations with cost larger than this value are ignored.
    tracks : List[track.Track]
        A list of predicted tracks at the current time step.
    detections : List[detection.Detection]
        A list of detections at the current time step.
    track_indices : List[int]
        List of track indices that maps rows in `cost_matrix` to tracks in `tracks` (see description above).
    detection_indices : List[int]
        List of detection indices that maps columns in `cost_matrix` to detections in `detections` (see description above).

    Returns
    -------
    (List[(int, int)], List[int], List[int])
        Returns a tuple with the following three entries:
        * A list of matched track and detection indices.
        * A list of unmatched track indices.
        * A list of unmatched detection indices.

    """
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))

    if len(detection_indices) == 0 or len(track_indices) == 0:
        return [], track_indices, detection_indices  # Nothing to match.

    # get cost matrix
    cost_matrix = distance_metric(tracks, detections, track_indices, detection_indices)
    cost_matrix[cost_matrix > max_distance] = max_distance + 1e-5    # assign distance>0.7 as 0.7

    # execute Hungarian algorithm，get matched indices pair，
    # row_indicies refer to tracks，column_indices refer to detections
    row_indices, col_indices = linear_assignment(cost_matrix)

    matches, unmatched_tracks, unmatched_detections = [], [], []
    # find unmatched detections
    for col, detection_idx in enumerate(detection_indices):
        if col not in col_indices:
            unmatched_detections.append(detection_idx)
    # find unmatched tracks
    for row, track_idx in enumerate(track_indices):
        if row not in row_indices:
            unmatched_tracks.append(track_idx)

    # loop over matched (track, detection) indices pair
    for row, col in zip(row_indices, col_indices):
        track_idx = track_indices[row]
        detection_idx = detection_indices[col]
        # treated as unmatched if cost > max_distance
        if cost_matrix[row, col] > max_distance:
            unmatched_tracks.append(track_idx)
            unmatched_detections.append(detection_idx)
        else:
            matches.append((track_idx, detection_idx))

    return matches, unmatched_tracks, unmatched_detections


def matching_cascade(distance_metric, max_distance, cascade_depth, tracks, detections, track_indices=None, detection_indices=None):
    """
    matching cascade (first do matching for recently confirmed track, then less recent tracks...)

    Parameters
    ----------
    distance_metric : a function, [List[Track], List[Detection], List[int], List[int]) --> 2D array
        Given a list of tracks and detections, a list of N track indices and M detection indices.
        The metric should return the 2D cost matrix (N, M),
        where element (i, j) is the cost between the i-th track and the j-th detection.
    max_distance : float
        Gating threshold of Mahala distance. Associations with cost > this value are ignored, default as 0.2
    cascade_depth: int
        The cascade depth, should be se to the maximum track age, default as 70
    tracks : List[track.Track]
        A list of predicted tracks at the current time step.
    detections : List[detection.Detection]
        A list of detections at the current time step.
    track_indices : Optional[List[int]]
        List of track indices that maps rows in `cost_matrix` to tracks in
        `tracks` (see description above). Defaults to all tracks.
    detection_indices : Optional[List[int]]
        List of detection indices that maps columns in `cost_matrix` to
        detections in `detections` (see description above). Defaults as all detections.

    Returns
    -------
    (List[(int, int)], List[int], List[int])
        Returns a tuple with the following three entries:
        * A list of matched track and detection indices.
        * A list of unmatched track indices.
        * A list of unmatched detection indices.
    """

    # initialise 2 lists: track_indices and detection_indices
    if track_indices is None:
        track_indices = list(range(len(tracks)))
    if detection_indices is None:
        detection_indices = list(range(len(detections)))

    unmatched_detections = detection_indices    # initialise unmatched detection set as all
    matches = []    # initialise matched detection set as None

    """matching cascade"""
    # first do matching for recently confirmed track, then less recent tracks...
    for level in range(cascade_depth):
        # break out of there is no detections
        if len(unmatched_detections) == 0:
            break

        # Select tracks by age
        track_indices_l = [
            k for k in track_indices
            if tracks[k].time_since_update == 1 + level
        ]
        # skip if no track at current level (age)
        if len(track_indices_l) == 0:
            continue

        # call Hungarian algorithm to do matching (cost matrix is cosine feature + Mahala distance)
        matches_l, _, unmatched_detections = \
            min_cost_matching(distance_metric, max_distance, tracks, detections, track_indices_l, unmatched_detections)

        # add matched ID pair [(track_id, detection_id)...] into matches
        matches += matches_l

    # set() removes redundancy, set()-set() returns difference set, i.e. unmatched tracks
    unmatched_tracks = list(set(track_indices) - set(k for k, _ in matches))

    return matches, unmatched_tracks, unmatched_detections


def gate_cost_matrix(kf, cost_matrix, tracks, detections, track_indices, detection_indices, gated_cost=INFTY_COST, only_position=False):
    """
    Apply Mahalanobias distance to modify the cost matrix of cosine feature distance
    Mahalanobias distance measures the bbox distance between Kalman predictions and YOLO detections.

    如果一个轨迹要去匹配两个外观特征非常相似的 detections，很容易出错；
    分别计算这两个 detections 与这个轨迹的马氏距离，并使用一个阈值gating_threshold进行限制，
    就可以将马氏距离较远的那个 detection 排除，从而减少错误的匹配

    @param kf: Kalman filter class instance
    @param cost_matrix: 2D array (num_tracks, num_detections), each element is the association cost
    @param tracks: a list of predicted tracks at the current time step
    @param detections: a list of detections at the current time step
    @param track_indices: a list of track indices in cost_matrix
    @param detection_indices: a list of detection indices in cost_matrix
    @param gated_cost: 1e5, associations with large M distance are set a very large value in the cost matrix
    @param only_position: if True, only the x, y position of the state distribution is considered in M distance
    @return:
        cost_matrix: 2D array, the modified cost matrix
    """

    gating_dim = 2 if only_position else 4    # dim of measurement space, default as 4

    # 马氏距离通过计算 detections 与平均轨迹位置的距离超过多少标准差来考虑 detections 的不确定性。
    # 通过从逆chi^2分布计算95%置信区间的阈值，排除可能性小的关联, 四维测量空间对应的马氏阈值为 9.4877
    gating_threshold = kalman_filter.chi2inv95[gating_dim]    # 9.4877
    measurements = np.asarray([detections[i].to_xyah() for i in detection_indices])    # 2D array (num_detections, 4)

    # compute the Mahala distance between Kalman predictions and detections
    # modify the cost of entries that have large distance
    for row, track_idx in enumerate(track_indices):
        track = tracks[track_idx]
        gating_distance = kf.gating_distance(track.mean, track.covariance, measurements, only_position)
        cost_matrix[row, gating_distance > gating_threshold] = gated_cost
    return cost_matrix
