# vim: expandtab:ts=4:sw=4
import numpy as np


class TrackState:
    """
    Enumeration type for the single target track state. Newly created tracks are
    classified as `tentative` until enough evidence has been collected. Then,
    the track state is changed to `confirmed`. Tracks that are no longer alive
    are classified as `deleted` to mark them for removal from the set of active
    tracks.

    """

    Tentative = 1
    Confirmed = 2
    Deleted = 3


class Track:
    """
    A single target track with state space `(x, y, a, h)` and associated
    velocities, where `(x, y)` is the center of the bounding box, `a` is the
    aspect ratio and `h` is the height.

    Parameters
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.
    max_age : int
        The maximum number of consecutive misses before the track state is
        set to `Deleted`.
    feature : Optional[ndarray]
        Feature vector of the detection this track originates from. If not None,
        this feature is added to the `features` cache.

    Attributes
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    hits : int
        Total number of measurement updates.
    age : int
        Total number of frames since first occurance.
    time_since_update : int
        Total number of frames since last measurement update.
    state : TrackState
        The current track state.
    features : List[ndarray]
        A cache of features. On each measurement update, the associated feature
        vector is added to this list.

    """

    def __init__(self, mean=None, covariance=None, track_id=None, n_init=None, max_age=None, detection=None, kf=None):

        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0

        self.state = TrackState.Tentative
        self.features = []
        if detection.feature is not None:
            self.features.append(detection.feature)

        self._n_init = n_init
        self._max_age = max_age
        self.cls_id = detection.cls_id
        self.covariance = detection.confidence
        # self.utm_pos = detection.utm_pos
        self.confidence = detection.confidence
        self.kf_utm = kf    # kalman filter
        self.bbox_width = detection.tlwh[2]
        self.bbox_height = detection.tlwh[3]
        self.detection_xyah = detection.to_xyah()
        self.cov_eigenvalues, self.cov_eigenvectors = self.get_gated_area(covariance[:2, :2], dims=2)
        self.xyz_rel2cam = detection.xyz_rel2cam
        self.imu_utm = None

    def to_tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
        width, height)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    def to_tlbr(self):
        """Get current position in bounding box format `(min x, miny, max x,
        max y)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret

    def utm_to_bbox_tlbr(self, cam2world):
        r0, c0, height = cam2world.convert_utm_coordinates_to_bbox_center(self.mean[:3])

        aspect_ratio = self.detection_xyah[-2]
        width = height * aspect_ratio

        xmin = c0 - width/2
        ymin = r0 - height/2
        xmax = c0 + width/2
        ymax = r0 + height/2

        return np.array([xmin,ymin, xmax,ymax])

    def predict(self, cam2world):
        """Propagate the state distribution to the current time step using a
        Kalman filter prediction step.

        The predicted position in predicted only if target was detected in prev frame. otherwise, target is occluded,
        and only the uncertainty is progressed by another prediction step

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.

        """
        time_stamp = cam2world.telemetry["timestamp"][0]
        if self.time_since_update == 0:
            self.mean, self.covariance = self.kf_utm.predict(time_stamp)
        else:
            _, self.covariance = self.kf_utm.predict(time_stamp)
        self.age += 1
        self.time_since_update += 1

        self.in_cam_FOV = cam2world.is_utm_coordinates_in_cam_FOV(self.mean[:3])

        # calculating gating area:
        self.cov_eigenvalues, self.cov_eigenvectors = self.get_gated_area(self.covariance[:2, :2])

    def update(self, detection, cam2world=None):
        """Perform Kalman filter measurement update step and update the feature
        cache.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.
        detection : Detection
            The associated detection.

        """
        self.mean, self.covariance = self.kf_utm.update_by_measurment(detection.utm_pos)
        self.features.append(detection.feature)
        self.confidence = detection.confidence
        self.cls_id = detection.cls_id
        self.xyz_rel2cam = detection.xyz_rel2cam
        self.detection_xyah = detection.to_xyah()

        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed

    def mark_missed(self):
        """Mark this track as missed (no association at the current time step).
        """
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted

    def is_tentative(self):
        """Returns True if this track is tentative (unconfirmed).
        """
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        """Returns True if this track is confirmed."""
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        """Returns True if this track is dead and should be deleted."""
        return self.state == TrackState.Deleted

    @staticmethod
    def get_gated_area(covarianceMat, dims=2):
        """
        calculate eigen vectors and eigen values and then multiply by eigenvalues by chi2inv95 of K dims
        to get ellipsoid shape
        Args:
            covarianceMat:

        Returns:

        """
        assert covarianceMat.shape == (dims, dims), "covariance shape != (2,2)"

        eigenvalues, eigenvectors = np.linalg.eig(covarianceMat)
        return eigenvalues, eigenvectors
