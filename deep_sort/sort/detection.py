# vim: expandtab:ts=4:sw=4
import numpy as np


class Detection(object):
    """
    This class represents a bounding box detection in a single image.

    Parameters
    ----------
    tlwh : array_like
        Bounding box in format `(x, y, w, h)`.
    confidence : float
        Detector confidence score.
    feature : array_like
        A feature vector that describes the object contained in this image.

    Attributes
    ----------
    tlwh : ndarray
        Bounding box in format `(top left x, top left y, width, height)`.
    confidence : ndarray
        Detector confidence score.
    feature : ndarray | NoneType
        A feature vector that describes the object contained in this image.

    utm_coordinates: ndarray
        UTM coordinates in format '(lat, long, height)'

    """

    def __init__(self, tlwh, confidence, cls_id, feature):
        self.tlwh = np.asarray(tlwh, dtype=np.float)
        self.confidence = float(confidence)
        # each cls is orthogonal for other cls for cosine similarity
        feature = np.asarray(feature, dtype=np.float32)
        self.feature = feature
        self.cls_id = cls_id
        self.utm_pos = []
        self.timestamp = []
        self.xyz_rel2cam = []

    def to_tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    def to_xyah(self):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def update_positions_using_telemetry(self, cam2world):
        tlbr = self.to_tlbr()
        self.utm_pos = cam2world.convert_bbox_tlbr_to_utm(tlbr)
        self.xyz_rel2cam = cam2world.convert_bbox_tlbr_to_xyz_rel2cam(tlbr)
        self.timestamp = cam2world.telemetry["timestamp"][0]

    def project_utm_to_bbox_tlwh(self, cam2world, utm_pos):
        row, col, height = cam2world.convert_utm_coordinates_to_bbox_center(utm_pos)
        bbox_tlwh = self.tlwh.copy()
        bbox_tlwh[0] = col - bbox_tlwh[2]/2
        bbox_tlwh[1] = row - bbox_tlwh[3]/2

        return bbox_tlwh