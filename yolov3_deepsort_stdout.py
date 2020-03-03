import os
import cv2
import time
import argparse
import torch
import numpy as np
from distutils.util import strtobool

from detector import build_detector
from deep_sort import build_tracker
from utils.draw import draw_boxes
from utils.parser import get_config
from utils.camera2world import Cam2World


class Tracker(object):
    def __init__(self, cfg, args):
        self.cfg = cfg
        self.args = args
        use_cuda = args.use_cuda and torch.cuda.is_available()
        if not use_cuda:
            raise UserWarning("Running in cpu mode!")

        if args.display:
            cv2.namedWindow("test", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("test", args.display_width, args.display_height)
   
        self.cam2world = Cam2World(self.args.img_width, self.args.img_height,
                                   self.args.thetaX, self.args.thetaY)
        self.vdo = cv2.VideoCapture()
        self.detector = build_detector(cfg, use_cuda=use_cuda)
        self.deepsort = build_tracker(cfg, use_cuda=use_cuda)
        self.class_names = self.detector.class_names
        self.bbox_xyxy = []
        self.identities = []
        self.target_id = []
        print("init OK")

    def select_target(self, event, col, row, flag, params):
        if event == cv2.EVENT_LBUTTONDBLCLK:  # right mouse click
            print('selected pixel (row, col) = ({},{})'.format(row, col))
            self.target_init_pos = [row, col]

            # selecting track id
            for ind, bbox in enumerate(self.bbox_xyxy):
                if bbox[0] <= col <= bbox[2] and bbox[1] <= row <= bbox[3]:
                    self.target_id = self.identities[ind]
                    print("selected target {}".format(self.target_id))

        else:
            pass

    def __enter__(self):
        assert os.path.isfile(self.args.VIDEO_PATH), "Error: path error"
        self.vdo.open(self.args.VIDEO_PATH)
        self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if self.args.save_path:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.writer = cv2.VideoWriter(self.args.save_path, fourcc, 20, (self.im_width, self.im_height))

        assert self.vdo.isOpened()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)

    def DeepSort(self, img_filename, target_cls):

        err_flag = ''
        im = cv2.imread(img_filename)
        if im is None:
            err_flag = "error loading filename"
            print(err_flag)
            return [], [], [], err_flag

               # do detection
        bbox_xywh, cls_conf, cls_ids = self.detector(im)
        outputs = []
        detections = []
        detections_conf = []
        target_xyz = []

        if bbox_xywh is not None:
            # select person class
            for cls in target_cls:
                try:
                    mask += cls_ids == cls
                except Exception:
                    mask = cls_ids == cls

            bbox_xywh = bbox_xywh[mask]
            bbox_xywh[:, 3:] *= 1.2  # bbox dilation just in case bbox too small
            cls_conf = cls_conf[mask]

            # do tracking
            outputs, detections, detections_conf = self.deepsort.update(bbox_xywh, cls_conf, im)

            # calculate object distance and direction from camera
            target_xyz = []
            for i, target in enumerate(outputs):
                target_xyz.append(self.cam2world.obj_world_xyz(x1=target[0],
                                                               y1=target[1],
                                                               x2=target[2],
                                                               y2=target[3],
                                                               obj_height_meters=self.args.target_height))

            for i, target in enumerate(outputs):
                id = target[-1]
                print('\t\t target [{}] \t ({},{},{},{})\t conf {:.2f}\t position: ({:.2f}, {:.2f}, {:.2f})[m]'
                      .format(id, target[0], target[1], target[2], target[3], detections_conf[i],
                              target_xyz[i][0], target_xyz[i][1], target_xyz[i][2]))

        return outputs, detections, detections_conf, target_xyz

def parse_args():
    parser = argparse.ArgumentParser()
    #parser.add_argument("VIDEO_PATH", type=str)
    parser.add_argument("--config_detection", type=str, default="./configs/yolov3.yaml")
    parser.add_argument("--config_deepsort", type=str, default="./configs/deep_sort.yaml")
    parser.add_argument("--ignore_display", dest="display", action="store_false", default=True)
    parser.add_argument("--display_width", type=int, default=800)
    parser.add_argument("--display_height", type=int, default=600)
    parser.add_argument("--save_path", type=str, default="./demo/demo.avi")
    parser.add_argument("--cpu", dest="use_cuda", action="store_false", default=True)
    parser.add_argument("--target_cls", type=str, default='0', help='coco dataset labels to track')
    parser.add_argument("--img-width", type=int, default=1280,
                        help='img width in pixels')
    parser.add_argument("--img-height", type=int, default=720,
                        help='img height in pixels')
    parser.add_argument("--thetaY", type=float, default=19.0,
                        help='angular camera FOV in vertical direction. [deg]')
    parser.add_argument("--thetaX", type=float, default=62.0,
                        help='angular camera FOV in horizontal direction. [deg]')
    parser.add_argument("--target-height", type=float, default=1.8,
                        help='tracked target height in [meters]')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = get_config()
    cfg.merge_from_file(args.config_detection)
    cfg.merge_from_file(args.config_deepsort)

    torch.device("cuda:1")

    tracker = Tracker(cfg, args)
    tracker.run()
