from os.path import dirname, join

import numpy as np
import torch

from detector import build_detector
from deep_sort.deep_sort import DeepSort
from utils.parser import get_config
from utils.camera2world import Cam2World
from utils.log_manager.deepsort_logger import DeepSortLogger

def get_merged_config():
    cfg = get_config()
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

        self.cam2world = Cam2World(self.width, self.height,
                                   self.camera_fov_x,
                                   self.target_height_m,
                                   cam_yaw_pitch_roll=self.cfg.CAMERA_YAW_PITCH_ROLL,
                                   cam_pos=self.cfg.CAMERA2IMU_DIST)

        self.cls_dict = {0: 'person', 2: 'car', 7: 'car'}

        # self.detector = build_detector(self.cfg, use_cuda=use_cuda)
        self.deepsort = DeepSort(self.cfg, use_cuda=use_cuda)
        self.class_names = self.deepsort.detector.class_names
        self.bbox_xyxy = []
        self.identities = []
        self.target_id = []

        # configuring logger:
        self.logger = DeepSortLogger()
        self.frame = -1

    def run_tracking(self, im, telemetry):
        self.frame += 1

        # update cam2world functions with new telemetry:
        self.cam2world.digest_new_telemetry(telemetry)

        # do detection:
        detections = self.deepsort.detect(im)

        # udpate detections with utm coordinates using cam2world
        for detection in detections:
            detection.update_positions_using_telemetry(self.cam2world)

        # associate tracks with detections
        tracks = self.deepsort.track(detections, cam2world=self.cam2world)

        self.logger.write(self.frame, tracks)

        return tracks, detections