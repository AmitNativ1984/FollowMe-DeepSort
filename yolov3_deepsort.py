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
from utils.draw import draw_boxes
from utils.parser import get_config
from utils.camera2world import Cam2World

from pysemanticsegmentation import core as segmentation_core

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
        self.cls_dict = {0: 'person', 2: 'car', 7: 'car'}
        self.vdo = cv2.VideoCapture()
        self.detector = build_detector(cfg, args, use_cuda=use_cuda)
        self.deepsort = build_tracker(cfg, use_cuda=use_cuda)
        self.class_names = self.detector.class_names
        self.bbox_xyxy = []
        self.target_id = []

        # defining segmentor
        # self.ss_person_model = segmentation_core.Model(graph_filename=self.cfg.PERSON_SEGMENTATION.GRAPH_CKPT, cfg=self.cfg.PERSON_SEGMENTATION)
        self.ss_vehicle_model = segmentation_core.Model(graph_filename=self.cfg.VEHICLE_SEGMENTATION.GRAPH_CKPT, cfg=self.cfg.VEHICLE_SEGMENTATION)
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
        target_cls = list(self.cls_dict.keys())
        target_name = list(self.cls_dict.values())
        while self.vdo.grab():
            start = time.time()
            _, ori_im = self.vdo.retrieve()
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

    def deep_sort_from_pointer(self, im_ptr, target_cls):
        ptr_type = POINTER(c_uint8)
        num_pixels = self.args.img_height * self.args.img_width
        num_bytes = num_pixels * 3
        im_contiguous = np.ctypeslib.as_array(cast(im_ptr, ptr_type), shape=(num_bytes,))

        return self.DeepSort(im_contiguous, target_cls)

    def DeepSort(self, im, target_cls):
        start_time = time.time()
        im = im.reshape(self.args.img_height, self.args.img_width, 3)
        # do detection
        bbox_xywh, cls_conf, cls_ids = self.detector(im)    # get all detections from image
        tracks = []
        detections = []

        # select supported classes
        mask = np.isin(cls_ids[0], list(self.cls_dict.keys()))
        bbox_xywh = bbox_xywh[0][mask]
        cls_conf = cls_conf[0][mask]
        cls_ids = cls_ids[0][mask]

        if len(cls_ids) > 0:
            tracks, detections = self.deepsort.update(bbox_xywh, cls_conf, cls_ids, im)
            # calculate object distance and direction from camera
            for i, track in enumerate(tracks):
                track.to_xyz(self.cam2world, obj_height_meters=self.args.target_height)
                # todo: create batch of detections
                x0, y0, x1, y1 = track.to_tlbr().astype(np.int)
                x0, x1 = np.clip([x0, x1], a_min=0, a_max=im.shape[1])
                y0, y1 = np.clip([y0, y1], a_min=0, a_max=im.shape[0])

                w = x1 - x0
                h = y1 - y0
                org_box = im[y0:y1, x0:x1]
                box, box_org_W, box_org_H = preprocess_segmentation(org_box,
                                                                    self.cfg.VEHICLE_SEGMENTATION.INPUT_SIZE_WIDTH,
                                                                    self.cfg.VEHICLE_SEGMENTATION.INPUT_SIZE_HEIGHT)
                mask=self.ss_vehicle_model.run(PIL.Image.fromarray(box))
                out_mask = postprocess_segmentation(np.asarray(mask[1]), box_org_W, box_org_H)
                if track.cls_id == 0.:
                    mask_id = 10
                else:
                    mask_id = 11

                out_mask[out_mask != mask_id] = 0
                out_mask[out_mask == mask_id] = 1
                track.mask = np.zeros_like(org_box[...,0])
                track.mask = out_mask[:h, :w]



            # todo: semantic segmenation of track bboxes

        print('process time: {}'.format(time.time() - start_time))
        return tracks, detections

def preprocess_segmentation(bbox_img, W0, H0):
    H, W, C = bbox_img.shape
    # padding_frame = (np.max([W // W0, H // H0]) + 1) * 256
    padding_frame = (np.max([W, H]))
    resize_factor = W0 / padding_frame
    bbox_img = cv2.copyMakeBorder(bbox_img, 0, np.abs(padding_frame-H), 0, np.abs(padding_frame-W), cv2.BORDER_CONSTANT, value=[0,0,0])
    padded_img_shape = bbox_img.shape
    return bbox_img, padded_img_shape[0], padded_img_shape[1]

def postprocess_segmentation(mask, box_org_W, box_org_H):
    mask = cv2.resize(mask, (box_org_W, box_org_H), interpolation=cv2.INTER_NEAREST)
    return mask


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, default='')
    parser.add_argument("--config_detection", type=str, default="./configs/yolov3_probot_ultralytics.yaml")
    parser.add_argument("--config_deepsort", type=str, default="./configs/deep_sort.yaml")
    parser.add_argument("--config_segmentation", type=str, default="./configs/segmentation.yaml")
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
    cfg.merge_from_file(args.config_segmentation)

    torch.set_num_threads(cfg.NUM_CPU_CORES)
    print("Using {} CPU cores".format(torch.get_num_threads()))

    with Tracker(cfg, args) as tracker:
        tracker.run()
