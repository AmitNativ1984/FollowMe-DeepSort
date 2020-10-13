import os
import cv2
import time
import argparse
import torch
import numpy as np
import ctypes
import shutil
from ctypes import POINTER, c_uint8, cast

from detector import build_detector
from deep_sort import build_tracker
from utils.draw import draw_boxes
from utils.parser import get_config
from utils.camera2world import Cam2World
from utils.datasets import LoadImages
from torch.utils.data import DataLoader

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
        self.cls_dict = {0: 'person', 2: 'vehicle', 7: 'vehicle'}
        self.dataset = LoadImages(self.args.root_dir)

        self.dataloader = DataLoader(self.dataset, batch_size=1,
                        shuffle=False)

        self.vdo = cv2.VideoCapture()
        self.detector = build_detector(cfg, use_cuda=use_cuda)
        self.deepsort = build_tracker(cfg, cam2world=self.cam2world,
                                      obj_height_meters=self.args.target_height,
                                      use_cuda=use_cuda)
        self.class_names = self.detector.class_names
        self.bbox_xyxy = []
        self.target_id = []

        self.output_logs_path = "./Logs"
        # cleanup prev records:
        try:
            shutil.rmtree(self.output_logs_path)
        except:
            pass

        os.makedirs(self.output_logs_path, exist_ok=True)

    def __enter__(self):
        assert os.path.isdir(self.args.root_dir), "Error: path error"

        self.im_width = 1280
        self.im_height = 720

        if self.args.save_path:
            fourcc =  cv2.VideoWriter_fourcc(*'XVID')
            self.writer = cv2.VideoWriter(self.args.save_path, fourcc, 25, (self.im_width, self.im_height))

        return self

    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)

    def run(self):
        frame = 0
        target_cls = list(self.cls_dict.keys())
        target_name = list(self.cls_dict.values())
        for frame, ori_im in enumerate(self.dataloader):
            print("frame: %d" % (frame))
            start = time.time()
            ori_im = np.array(ori_im).squeeze(0)
            im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)

            tracks, detections = self.DeepSort(im, target_cls)

            # draw boxes for visualization
            if len(detections) > 0 and args.debug_mode and args.display:
                DetectionsBBOXES = np.array([detections[i].tlwh for i in range(np.shape(detections)[0])])
                DetectionsBBOXES[:, 2] += DetectionsBBOXES[:, 0]
                DetectionsBBOXES[:, 3] += DetectionsBBOXES[:, 1]
                bbox_xyxy = DetectionsBBOXES[:, :4]
                ori_im = draw_boxes(ori_im, bbox_xyxy)

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
                    cls_names.append(self.cls_dict[track.cls_id])

                    with open(os.path.join(self.output_logs_path, "tracks_log.txt"), "+a") as log:
                        line = "{},{},{},{},{}\n".format(frame, track.cls_id, track.confidence,
                                                          track.to_tlbr(), np.array(track.xyz_pos))
                        log.write(line)

                    with open(os.path.join(self.output_logs_path, "track_{}.txt".format(track.track_id)), "+a") as log:
                        line = "{},{},{},{},{}\n".format(frame, track.cls_id, track.confidence,
                                                          track.to_tlbr(), np.array(track.xyz_pos))
                        log.write(line)
                ori_im = draw_boxes(ori_im, bbox_xyxy, identities, target_id=self.target_id, confs=confs,
                                    target_xyz=xyz_pos, cls_names=cls_names)

            if args.display:
                cv2.imshow("test", ori_im)
                key = cv2.waitKey(1)

            if self.args.save_path:
                self.writer.write(ori_im)

            frame += 1

        if self.args.save_path:
            self.writer.release()

    def deep_sort_from_pointer(self, im_ptr, target_cls):
        ptr_type = POINTER(c_uint8)
        num_pixels = self.args.img_height * self.args.img_width
        num_bytes = num_pixels * 3
        im_contiguous = np.ctypeslib.as_array(cast(im_ptr, ptr_type), shape=(num_bytes,))

        return self.DeepSort(im_contiguous, target_cls)

    def DeepSort(self, im, target_cls):
        im = im.reshape(self.args.img_height, self.args.img_width, 3)
        # do detection
        bbox_xywh, cls_conf, cls_ids = self.detector(im)    # get all detections from image
        tracks = []
        detections = []

        # select person class
        mask = np.isin(cls_ids[0], list(self.cls_dict.keys()))
        bbox_xywh = bbox_xywh[0][mask]
        cls_conf = cls_conf[0][mask]
        cls_ids = cls_ids[0][mask]
        cls_ids = np.array([2. if id != 0 else id for id in cls_ids])

        if len(cls_ids) > 0:
            tracks, detections = self.deepsort.update(bbox_xywh, cls_conf, cls_ids, im)

        return tracks, detections

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", type=str, default='')
    # parser.add_argument("--config_detection", type=str, default="./configs/yolov3_probot_ultralytics.yaml")
    parser.add_argument("--config_deepsort", type=str, default="./configs/deep_sort.yaml")
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
    # cfg.merge_from_file(args.config_detection)
    cfg.merge_from_file(args.config_deepsort)

    torch.set_num_threads(cfg.NUM_CPU_CORES)
    print("Using {} CPU cores".format(torch.get_num_threads()))

    with Tracker(cfg, args) as tracker:
        tracker.run()
