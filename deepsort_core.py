import time
import numpy as np

class DeepSort(object):
    def __init__(self, detector, deepsort_tracker, segmentor, cls_dict, cam2world, target_height):
        self.detector = detector
        self.deepsort_tracker = deepsort_tracker
        self.segmentor = segmentor
        self.cls_dict = cls_dict
        self.cam2world = cam2world
        self.target_height = target_height

    def detect_and_track(self, im):

        # do detection
        bbox_xywh, cls_conf, cls_ids = self.detector(im)  # get all detections from image
        tracks = []
        detections = []

        # select supported classes
        mask = np.isin(cls_ids[0], list(self.cls_dict.keys()))
        bbox_xywh = bbox_xywh[0][mask]
        cls_conf = cls_conf[0][mask]
        cls_ids = cls_ids[0][mask]

        if len(cls_ids) > 0:
            tracks, detections = self.deepsort_tracker.update(bbox_xywh, cls_conf, cls_ids, im)
            tracks = self.segmentor.segment_bboxes(tracks, im)
            tracks = self.deepsort_tracker.calculate_track_xyz_pos(self.cam2world, tracks, self.target_height)

        return tracks, detections
