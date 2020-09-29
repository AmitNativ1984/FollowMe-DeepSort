import time
import numpy as np

class DeepSort(object):
    def __init__(self, detector, deepsort_tracker, cls_dict, cam2world, target_height):
        self.detector = detector
        self.deepsort_tracker = deepsort_tracker
        self.cls_dict = cls_dict
        self.cam2world = cam2world
        self.target_height = target_height

    def detect(self, img):

        # do detection
        bbox_xywh, cls_conf, cls_ids = self.detector(img)  # get all detections from image

        detections = []
        # select supported classes
        mask = np.isin(cls_ids[0], list(self.cls_dict.keys()))
        bbox_xywh = bbox_xywh[0][mask]
        cls_conf = cls_conf[0][mask]
        cls_ids = cls_ids[0][mask]

        detections = self.deepsort_tracker.detect(bbox_xywh, cls_conf, cls_ids, img)
        return detections

    def track(self, detections, im):

        tracks = self.deepsort_tracker.update_tracks(detections)
        tracks = self.deepsort_tracker.calculate_track_xyz_pos(self.cam2world, tracks, self.target_height)

        return tracks
