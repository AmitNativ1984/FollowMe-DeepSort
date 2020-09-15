import time
import numpy as np

def DeepSort(im, detector, deepsort_tracker, segmentor, cls_dict, cam2world, target_height):

    # do detection
    bbox_xywh, cls_conf, cls_ids = detector(im)  # get all detections from image
    tracks = []
    detections = []

    # select supported classes
    mask = np.isin(cls_ids[0], list(cls_dict.keys()))
    bbox_xywh = bbox_xywh[0][mask]
    cls_conf = cls_conf[0][mask]
    cls_ids = cls_ids[0][mask]

    if len(cls_ids) > 0:
        tracks, detections = deepsort_tracker.update(bbox_xywh, cls_conf, cls_ids, im)
        tracks = segmentor.segment_bboxes(tracks, im)
        tracks = deepsort_tracker.calculate_track_xyz_pos(cam2world, tracks, target_height)

    return tracks, detections
