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

# from pysemanticsegmentation import core as segmentation_core
from segmentor.pytorch_deeplab.infer_bbox import InferBoundingBox

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
        self.detector = build_detector(cfg, args, use_cuda=use_cuda)
        self.deepsort = build_tracker(cfg, use_cuda=use_cuda)
        self.class_names = self.detector.class_names
        self.bbox_xyxy = []
        self.target_id = []

        # defining segmentor
        self.segmentor = InferBoundingBox(cfg.SEGMENTOR)
        self.segmentor.define_model()
        self.segmentor.load_model()

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
            tracks = self.segment_bboxes(tracks, im)
            tracks = self.calculate_track_xyz_pos(tracks)


        print('process time: {}'.format(time.time() - start_time))
        return tracks, detections

    def calculate_track_xyz_pos(self, tracks):
        for _, track in enumerate(tracks):
            track.to_xyz(self.cam2world, obj_height_meters=self.args.target_height)

        return tracks

    def segment_bboxes(self, tracks, im):

        bbox = [None] * len(tracks)
        padded_bbox_shape = [None] * len(tracks)
        org_bbox_shape = [None] * len(tracks)
        bbox_resize_factor = [None] * len(tracks)
        bbox_cls_id = [None] * len(tracks)

        # batching all tracks for inference:
        for i, track in enumerate(tracks):

            x0, y0, x1, y1 = track.to_tlbr().astype(np.int)
            x0, x1 = np.clip([x0, x1], a_min=0, a_max=im.shape[1])
            y0, y1 = np.clip([y0, y1], a_min=0, a_max=im.shape[0])

            org_box = im[y0:y1, x0:x1]
            resized_bbox_tensor, padded_shape, org_shape, resize_factor = \
                self.preprocess_segmentation(org_box, self.cfg.SEGMENTOR.INPUT_SIZE_WIDTH, self.cfg.SEGMENTOR.INPUT_SIZE_HEIGHT)

            bbox[i] = resized_bbox_tensor
            padded_bbox_shape[i] = padded_shape
            org_bbox_shape[i] = org_shape
            bbox_resize_factor[i] = resize_factor
            bbox_cls_id[i] = track.cls_id


        if len(bbox) != 0:
            bbox_tensor = torch.cat(bbox).view(-1, 3, self.cfg.SEGMENTOR.INPUT_SIZE_HEIGHT, self.cfg.SEGMENTOR.INPUT_SIZE_WIDTH)
            bbox_tensor = bbox_tensor.cuda()
            org_bbox_shape = np.stack(org_bbox_shape, axis=0)

            raw_pred_mask = self.segmentor.infer(bbox_tensor)
            tracks = self.postprocess_segmentation(tracks, raw_pred_mask, bbox_cls_id, padded_bbox_shape, org_bbox_shape)

        return tracks

    def preprocess_segmentation(self, bbox_img, W0, H0):
        H, W, C = bbox_img.shape
        org_bbox_shape = (H, W)
        padding_frame = (np.max([W, H]))
        resize_factor = W0 / padding_frame
        padded_bbox_img = cv2.copyMakeBorder(bbox_img, 0, np.abs(padding_frame-H), 0, np.abs(padding_frame-W), cv2.BORDER_CONSTANT, value=[0,0,0])
        padded_bbox_shape = padded_bbox_img.shape[:2]
        resized_bbox_image = cv2.resize(padded_bbox_img, (H0, W0))
        resized_bbox_tensor = InferBoundingBox.preprocess()(resized_bbox_image)

        return resized_bbox_tensor, padded_bbox_shape, org_bbox_shape, resize_factor

    def postprocess_segmentation(self, tracks, raw_pred_mask, bbox_cls_id, padded_bbox_shape, org_bbox_shape):

        for idx in range(raw_pred_mask.shape[0]):
            mask = cv2.resize(raw_pred_mask[idx, ...], padded_bbox_shape[idx], interpolation=cv2.INTER_NEAREST)
            mask = mask[:org_bbox_shape[idx][0], :org_bbox_shape[idx][1]]
            segmentation_cls_id = self.segmentor.labels_decoder().cls_id_dict[self.cls_dict[bbox_cls_id[idx]]]
            mask[mask != segmentation_cls_id] = 0
            mask[mask == segmentation_cls_id] = 1

            tracks[idx].mask = mask


        return tracks

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
