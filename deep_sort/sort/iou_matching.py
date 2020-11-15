# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from . import linear_assignment


def iou(bbox, candidates):
    """Computer intersection over union.

    Parameters
    ----------
    bbox : ndarray
        A bounding box in format `(top left x, top left y, width, height)`.
    candidates : ndarray
        A matrix of candidate bounding boxes (one per row) in the same format
        as `bbox`.

    Returns
    -------
    ndarray
        The intersection over union in [0, 1] between the `bbox` and each
        candidate. A higher score means a larger fraction of the `bbox` is
        occluded by the candidate.

    """
    bbox_tl, bbox_br = bbox[:2], bbox[:2] + bbox[2:]
    candidates_tl = candidates[:, :2]
    candidates_br = candidates[:, :2] + candidates[:, 2:]

    tl = np.c_[np.maximum(bbox_tl[0], candidates_tl[:, 0])[:, np.newaxis],
               np.maximum(bbox_tl[1], candidates_tl[:, 1])[:, np.newaxis]]
    br = np.c_[np.minimum(bbox_br[0], candidates_br[:, 0])[:, np.newaxis],
               np.minimum(bbox_br[1], candidates_br[:, 1])[:, np.newaxis]]
    wh = np.maximum(0., br - tl)

    area_intersection = wh.prod(axis=1)
    area_bbox = bbox[2:].prod()
    area_candidates = candidates[:, 2:].prod(axis=1)
    return area_intersection / (area_bbox + area_candidates - area_intersection)


def iou_cost(tracks, detections, track_indices=None,
             detection_indices=None, cam2world=None):
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

    detections_tlwh = np.array(([], [], [], [])).transpose()
    detections_cls_ids = []
    detections_utm = np.array([[], [], []])
    for det_ind in detection_indices:
        detections_tlwh = np.vstack((detections_tlwh, detections[det_ind].tlwh))
        detections_cls_ids.append(detections[det_ind].cls_id)
        detections_utm = np.hstack((detections_utm, detections[det_ind].utm_pos))

    for row, track_idx in enumerate(track_indices):
        if tracks[track_idx].time_since_update > 20 or not tracks[track_idx].in_cam_FOV:
            cost_matrix[row, :] = linear_assignment.INFTY_COST
            continue

        r0, c0, h = cam2world.convert_utm_coordinates_to_bbox_center(tracks[track_idx].mean[:3])
        xmin, ymin, xmax, ymax = np.array([c0[0] - tracks[track_idx].bbox_width / 2, r0[0] - tracks[track_idx].bbox_height / 2,
                         c0[0] + tracks[track_idx].bbox_width / 2 + 1, r0[0] + tracks[track_idx].bbox_height / 2 + 1])

        bbox = np.array([xmin, ymin, xmax-xmin, ymax-ymin])

        cost_matrix[row, :] = 1. - iou(bbox, detections_tlwh)

        # verify iou matches only on same class
        cost_matrix[row, detections_cls_ids != tracks[track_idx].cls_id] = linear_assignment.INFTY_COST

    # handle partial occlusions: verify iou matches only inside area of confusion (only for confirmed tracks)
    if tracks[track_idx].is_confirmed:
        cost_matrix = linear_assignment.gate_cost_matrix(cost_matrix, tracks, detections,
                                                         track_indices, detection_indices,
                                                         gated_cost=linear_assignment.INFTY_COST,
                                                         only_position=True)

    return cost_matrix
