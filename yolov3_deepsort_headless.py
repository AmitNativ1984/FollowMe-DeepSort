import os
from os.path import dirname, join

import numpy as np
import torch

from detector import build_detector
from deep_sort import build_tracker
from utils.parser import get_config
from utils.camera2world import Cam2World

def get_merged_config():
    cfg = get_config()

    cfg.merge_from_file(join(dirname(__file__), 'configs', 'yolov3_probot_ultralytics.yaml'))
    cfg.merge_from_file(join(dirname(__file__), 'configs', 'deep_sort.yaml'))

    return cfg

class Tracker:
    def __init__(self, width, height, classes, safezone, camera_fov_x, target_height_m):
        # Set internal variables.
        self.width = width
        self.height = height
        self.classes = classes
        self.safezone = safezone
        self.camera_fov_x = camera_fov_x
        self.target_height_m = target_height_m
        
        self.cfg = get_merged_config()
        use_cuda = torch.cuda.is_available()
        if not use_cuda:
            raise UserWarning("Running in cpu mode!")

        self.cam2world = Cam2World(self.width, self.height, self.camera_fov_x)
        self.cls_dict = {0: 'person', 2: 'car', 7: 'car'}

        self.detector = build_detector(self.cfg, use_cuda=use_cuda)
        self.deepsort = build_tracker(self.cfg, use_cuda=use_cuda)
        self.class_names = self.detector.class_names
        self.bbox_xyxy = []
        self.identities = []
        self.target_id = []

    def DeepSort(self, im):
        # do detection
        bbox_xywh, cls_conf, cls_ids = self.detector(im)    # get all detections from image
        tracks = []
        detections = []

        # select person class
        mask = np.isin(cls_ids[0], list(self.cls_dict.keys()))
        bbox_xywh = bbox_xywh[0][mask]
        cls_conf = cls_conf[0][mask]
        cls_ids = cls_ids[0][mask]

        if len(cls_ids) > 0:
            tracks, detections = self.deepsort.update(bbox_xywh, cls_conf, cls_ids, im)
            
            # calculate object distance and direction from camera
            for i, track in enumerate(tracks):
                track.to_xyz(self.cam2world, obj_height_meters=self.target_height_m)

        return tracks, detections