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

    def run(self):
        frame = 0
        target_cls = [0, 2, 7]

        while True:
            img_filename = input()
            if img_filename == 'terminate':
                return

            ori_im, outputs, detections, err_flag = self.DeepSort(img_filename, target_cls)
            print('[detections] {}'.format(len(outputs)))
            for target in outputs:
                id = target[-1]
                print('target [{}], bbox: ({},{},{},{})'.format(id, target[0], target[1], target[2], target[3]))

            if len(outputs) > 0:
                bbox_xyxy = outputs[:, :4]
                identities = outputs[:, -1]
                ori_im = draw_boxes(ori_im, bbox_xyxy, identities, target_id=self.target_id)
                self.bbox_xyxy = bbox_xyxy
                self.identities = identities
            if self.args.display:
                cv2.imshow("test", ori_im)
                key = cv2.waitKey(1)
                cv2.setMouseCallback("test", self.select_target)

    def DeepSort(self, img_filename, target_cls):

        err_flag = ''
        im = cv2.imread(img_filename)
        if im is None:
            err_flag = "error loading filename"
            print(err_flag)
            return [], [], [], err_flag

        # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        # do detection
        bbox_xywh, cls_conf, cls_ids = self.detector(im)
        outputs = []
        detections = []
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
            outputs, detections = self.deepsort.update(bbox_xywh, cls_conf, im)
        return im, outputs, detections, err_flag


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("VIDEO_PATH", type=str)
    parser.add_argument("--config_detection", type=str, default="./configs/yolov3.yaml")
    parser.add_argument("--config_deepsort", type=str, default="./configs/deep_sort.yaml")
    parser.add_argument("--ignore_display", dest="display", action="store_false", default=True)
    parser.add_argument("--display_width", type=int, default=800)
    parser.add_argument("--display_height", type=int, default=600)
    parser.add_argument("--save_path", type=str, default="./demo/demo.avi")
    parser.add_argument("--cpu", dest="use_cuda", action="store_false", default=True)
    parser.add_argument("--target_cls", type=str, default='0', help='coco dataset labels to track')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = get_config()
    cfg.merge_from_file(args.config_detection)
    cfg.merge_from_file(args.config_deepsort)

    torch.device("cuda:1")

    tracker = Tracker(cfg, args)
    tracker.run()
