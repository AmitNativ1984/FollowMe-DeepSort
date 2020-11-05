# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from . import kalman_filter
from . import kalman_filter_xyz

UTM_TRACKING = True
from . import iou_matching
from . import linear_assignment
from . import utm_iou_matching as utm_dist_matching
from . import utm_linear_assignment

if UTM_TRACKING:
    from .track_xyz import Track
else:
    from .track import Track


class Tracker:
    """
    This is the multi-target tracker.

    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.

    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    kf : kalman_filter.KalmanFilter
        A Kalman filter to filter target trajectories in image space.
    tracks : List[Track]
        The list of active tracks at the current time step.

    """

    def __init__(self, metric, max_iou_distance=0.7, max_age=70, n_init=3):
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init

        self.kf = kalman_filter.KalmanFilter()
        self.kf_utm = kalman_filter_xyz.KalmanXYZ
        self.tracks = []
        self._next_id = 1

    def predict(self, camera2world=None):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            if UTM_TRACKING:
                track.predict(camera2world)
            else:
                track.predict(self.kf)

    def update(self, detections, cam2world=None):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
        # Run matching cascade. (hungarian algorithm)
        matches, unmatched_tracks, unmatched_detections = \
            self._match(detections, cam2world)

        # Update track set.
        for track_idx, detection_idx in matches:    # update tracks with matched detections
            if UTM_TRACKING:
                self.tracks[track_idx].update(detections[detection_idx])
            else:
                self.tracks[track_idx].update(self.kf, detections[detection_idx])
        for track_idx in unmatched_tracks:          # update as track with no detections. add +1 to missed frames. delete if necessary
            self.tracks[track_idx].mark_missed()
        for detection_idx in unmatched_detections:  # if detections are not associated with any track, initiate track and start counting frames
            self._initiate_track(detections[detection_idx])
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Update distance metric.
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features  # add feature vector to list
            targets += [track.track_id for _ in track.features] # associate target id with every feature vector
            track.features = []
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets)

    def _match(self, detections, cam2world):

        def gated_metric(tracks, dets, track_indices, detection_indices, cam2world=None):
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix = self.metric.distance(features, targets)
            # gating the cost matrix based on mahalanobis distance of track. this happens after every matching based on
            # features
            if UTM_TRACKING:
                cost_matrix = utm_linear_assignment.gate_cost_matrix(
                    cost_matrix, tracks, dets, track_indices,
                    detection_indices)
            else:
                cost_matrix = linear_assignment.gate_cost_matrix(
                    self.kf, cost_matrix, tracks, dets, track_indices,
                    detection_indices)

            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # Associate CONFIRMED tracks with new detections by matching cascade.
        # result is matched/unmatched tracks and unmatched detections
        if UTM_TRACKING:
            matches_a, unmatched_tracks_a, unmatched_detections = \
                utm_linear_assignment.matching_cascade(
                    gated_metric, self.metric.matching_threshold, self.max_age,
                    self.tracks, detections, confirmed_tracks)
        else:
            matches_a, unmatched_tracks_a, unmatched_detections = \
                linear_assignment.matching_cascade(
                    gated_metric, self.metric.matching_threshold, self.max_age,
                    self.tracks, detections, confirmed_tracks)

        # Associate already known tentative tracks with unmatched tracks as for iou test
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update == 1]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update != 1]

        # Associate unmatched tracks and unmatched detections using IoU
        if False:
            matches_b, unmatched_tracks_b, unmatched_detections = \
                utm_linear_assignment.min_cost_matching(
                    utm_dist_matching.iou_cost, self.max_iou_distance, self.tracks,
                    detections, iou_track_candidates, unmatched_detections)
        else:
            matches_b, unmatched_tracks_b, unmatched_detections = \
                linear_assignment.min_cost_matching(
                    iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                    detections, iou_track_candidates, unmatched_detections, cam2world)

        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection):
        if UTM_TRACKING:
            new_kf = kalman_filter_xyz.KalmanXYZ()
            mean, covariance = new_kf.initiate(detection.timestamp, detection.utm_pos)
            self.tracks.append(Track(
                kf=new_kf, mean=mean, covariance=covariance, track_id=self._next_id, n_init=self.n_init, max_age=self.max_age,
                detection=detection))
        else:
            mean, covariance = self.kf.initiate(detection.to_xyah())

            self.tracks.append(Track(
                mean=mean, covariance=covariance, track_id=self._next_id, n_init=self.n_init, max_age=self.max_age,
                detection=detection))

        self._next_id += 1
