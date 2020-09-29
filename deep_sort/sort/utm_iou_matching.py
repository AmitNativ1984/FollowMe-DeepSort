# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from . import linear_assignment


def utm_dist(utm_pos, candidates):
    """Computer intersection over union.

    Parameters
    ----------
    bbox : ndarray
        A bounding box in format `(top left x, top left y, width, height)`.
    candidates : ndarray
        A matrix of candidate utm positions, same format as  utm_pos.

    Returns
    -------
    ndarray
        distance in utm [meters] between utm position and candidates

    """
    dist = np.sqrt((utm_pos[0] - candidates[:, 0]) ** 2 + (utm_pos[1] - candidates[:, 1]) ** 2)
    return dist


def iou_cost(tracks, detections, track_indices=None,
             detection_indices=None):
    """An intersection over union distance metric.

    Parameters
    ----------
    tracks : List[deep_sort.track.Track]
        A list of tracks.
    detections : List[deep_sort.detection.Detection]
        A list of detections.
    track_indices : Optional[List[int]]
        A list of indices to tracks that should be matched. Defaults to
        all `tracks`.
    detection_indices : Optional[List[int]]
        A list of indices to detections that should be matched. Defaults
        to all `detections`.

    Returns
    -------
    ndarray
        Returns a cost matrix of shape
        len(track_indices), len(detection_indices) where entry (i, j) is
        `1 - iou(tracks[track_indices[i]], detections[detection_indices[j]])`.

    """
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))

    cost_matrix = np.zeros((len(track_indices), len(detection_indices)))
    for row, track_idx in enumerate(track_indices):
        if tracks[track_idx].time_since_update > 3:
            cost_matrix[row, :] = linear_assignment.INFTY_COST
            continue

        utm_pos = tracks[track_idx].mean
        candidates = np.asarray([detections[i].utm_pos for i in detection_indices]).squeeze(-1)
        cost_matrix[row, :] = utm_dist(utm_pos, candidates)
    return cost_matrix
