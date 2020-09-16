import os
import cv2
import time
import argparse
import torch
import numpy as np
import ctypes
from ctypes import POINTER, c_uint8, cast
import PIL
from detector import build_detector
from deep_sort import build_tracker
from segmentor import build_segmentor
from utils.draw import draw_boxes
from utils.parser import get_config
from utils.camera2world import Cam2World
from deepsort_core import DeepSort

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
                                   self.args.thetaX)
        self.cls_dict = {0: 'Person', 2: 'Vehicle', 7: 'Vehicle'}
        self.vdo = cv2.VideoCapture()
        self.detector = build_detector(cfg, use_cuda=use_cuda)
        self.deepsort = build_tracker(cfg, use_cuda=use_cuda)
        self.segmentor = build_segmentor(cfg, self.cls_dict)
        self.class_names = self.detector.class_names
        self.bbox_xyxy = []
        self.target_id = []

        self.DeepSort = DeepSort(detector=self.detector,
                                 deepsort_tracker=self.deepsort,
                                 segmentor=self.segmentor,
                                 cls_dict=self.cls_dict,
                                 cam2world=self.cam2world,
                                 target_height=self.args.target_height)


    def __enter__(self):
        assert os.path.isfile(self.args.video_path), "Error: path error"
        self.vdo.open(self.args.video_path)
        self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if self.args.save_path:
            fourcc =  cv2.VideoWriter_fourcc(*'XVID')
            self.writer = cv2.VideoWriter(self.args.save_path, fourcc, 10, (self.im_width,self.im_height))

        assert self.vdo.isOpened()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)

    def run(self):
        frame = 0
        while self.vdo.grab():
            start = time.time()
            _, ori_im = self.vdo.retrieve()
            im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)

            detections = self.DeepSort.detect(im)
            tracks = self.DeepSort.track(detections, im)

            # draw boxes for visualization
            if len(detections) > 0 and args.debug_mode and args.display:
                DetectionsBBOXES = np.array([detections[i].tlwh for i in range(np.shape(detections)[0])])
                DetectionsBBOXES[:, 2] += DetectionsBBOXES[:, 0]
                DetectionsBBOXES[:, 3] += DetectionsBBOXES[:, 1]
                bbox_xyxy = DetectionsBBOXES[:, :4]
                ori_im = draw_boxes(ori_im, bbox_xyxy)

                blue, green, red = cv2.split(ori_im)
                for detection in detections:
                    x0, y0, x1, y1 = detection.to_tlbr().astype(np.int)
                    x0, x1 = np.clip([x0, x1], a_min=0, a_max=im.shape[1])
                    y0, y1 = np.clip([y0, y1], a_min=0, a_max=im.shape[0])

                    h = y1 - y0
                    w = x1 - x0
                    blue[y0:y0 + h, x0:x0 + w][detection.mask != 0] = detection.mask[detection.mask != 0] * 255

                ori_im = cv2.merge((blue, green, red))

            if len(tracks) > 0:
                bbox_xyxy = []
                identities = []
                confs = []
                xyz_pos = []
                cls_names = []

                for track in tracks:
                    bbox_xyxy.append(track.to_tlbr())
                    identities.append(track.track_id)
                    confs.append(track.confidence)
                    xyz_pos.append(track.xyz_pos)
                    cls_names.append(track.cls_id)

                    blue, green, red = cv2.split(ori_im)
                    x0, y0, x1, y1 = track.to_tlbr().astype(np.int)
                    x0, x1 = np.clip([x0, x1], a_min=0, a_max=im.shape[1])
                    y0, y1 = np.clip([y0, y1], a_min=0, a_max=im.shape[0])

                    h = y1 - y0
                    w = x1 - x0
                    if track.cls_id == 0:
                        red[y0:y0+h, x0:x0+w][track.mask !=0] = track.mask[track.mask !=0] * 255
                    else:
                        green[y0:y0 + h, x0:x0 + w][track.mask != 0] = track.mask[track.mask != 0] * 255

                    ori_im = cv2.merge((blue, green, red))
                ori_im = draw_boxes(ori_im, bbox_xyxy, identities, target_id=self.target_id, confs=confs,
                                    target_xyz=xyz_pos, cls_names=cls_names)

            if args.display:
                cv2.imshow("test", ori_im)
                key = cv2.waitKey(1)

            if self.args.save_path:
                self.writer.write(ori_im)

            print("frame: %d"%(frame))
            frame += 1

        if self.args.save_path:
            self.writer.release()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, default='')
    parser.add_argument("--config_detection", type=str, default="./configs/yolov3_probot_ultralytics.yaml")
    parser.add_argument("--config_deepsort", type=str, default="./configs/deep_sort.yaml")
    # parser.add_argument("--config_segmentation", type=str, default="./configs/segmentation.yaml")
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--display_width", type=int, default=800)
    parser.add_argument("--display_height", type=int, default=600)
    parser.add_argument("--save_path", type=str, default="./demo/demo.avi")
    parser.add_argument("--cpu", dest="use_cuda", action="store_false", default=True)
    parser.add_argument("--target_cls", type=str, default='0', help='coco dataset labels to track')
    parser.add_argument("--img-width", type=int, default=1280,
                        help='img width in pixels')
    parser.add_argument("--img-height", type=int, default=720,
                        help='img height in pixels')
    parser.add_argument("--thetaX", type=float, default=77.04,
                        help='angular camera FOV in horizontal direction. [deg]')
    parser.add_argument("--target-height", type=float, default=1.8,
                        help='tracked target height in [meters]')
    parser.add_argument("--debug-mode", action="store_true", default=False)

    return parser.parse_args()

if __name__=="__main__":
    args = parse_args()
    cfg = get_config()
    cfg.merge_from_file(args.config_detection)
    cfg.merge_from_file(args.config_deepsort)

    torch.set_num_threads(cfg.NUM_CPU_CORES)
    print("Using {} CPU cores".format(torch.get_num_threads()))

    with Tracker(cfg, args) as tracker:
        tracker.run()
