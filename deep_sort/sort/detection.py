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

    """

    def __init__(self, tlwh, confidence, cls_id, feature, cam2world, obj_height_meters):
        self.tlwh = np.asarray(tlwh, dtype=np.float)
        self.confidence = float(confidence)
        # each cls is orthogonal for other cls for cosine similarity
        feature = np.asarray(feature, dtype=np.float32)
        self.feature = feature
        self.cls_id = cls_id
        self.xyz_pos = []
        self.cam2world = cam2world
        self.obj_height_meters = obj_height_meters

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

    def to_xyz(self):
        tlbr = self.to_tlbr()
        self.xyz_pos = self.cam2world.obj_world_xyz(x1=tlbr[0],
                                          y1=tlbr[1],
                                          x2=tlbr[2],
                                          y2=tlbr[3],
                                          obj_height_meters=self.obj_height_meters)

        return self.xyz_pos