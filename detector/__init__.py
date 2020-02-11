from .YOLOv3 import YOLOv3
from .yolov3.infer_img import YOLOv3 as yolov3_ultralytics
from .yolov3.detect import detect
from utils.parser import get_config
from .YOLOv3.cfg import parse_cfg
import argparse

__all__ = ['build_detector']

def build_detector(cfg, use_cuda, implementation='org'):
    if implementation=='org':
        return YOLOv3(cfg.YOLOV3.CFG, cfg.YOLOV3.WEIGHT, cfg.YOLOV3.CLASS_NAMES,
                        score_thresh=cfg.YOLOV3.SCORE_THRESH, nms_thresh=cfg.YOLOV3.NMS_THRESH,
                        is_xywh=True, use_cuda=use_cuda)
        # returns bbox.numpy(), cls_conf.numpy(), cls_ids.numpy()

    elif implementation == 'new':
        parser = argparse.ArgumentParser()
        parser.add_argument('--cfg', type=str, default=cfg.YOLOV3.CFG, help='*.cfg path')
        parser.add_argument('--names', type=str, default=cfg.YOLOV3.CLASS_NAMES, help='*.names path')
        parser.add_argument('--weights', type=str, default=cfg.YOLOV3.WEIGHT, help='path to weights file')
        parser.add_argument('--source', type=str, default='/home/amit/Data/Vehicle_Recordings/Ben_Shemen/14-01-2020/day/drive3-day/images/drive_cam0_1012.bmp',
                            help='source')  # input file/folder, 0 for webcam
        parser.add_argument('--output', type=str, default='output', help='output folder')  # output folder
        parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
        parser.add_argument('--conf-thres', type=float, default=cfg.YOLOV3.SCORE_THRESH, help='object confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=cfg.YOLOV3.NMS_THRESH, help='IOU threshold for NMS')
        parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
        parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
        parser.add_argument('--device', default='0', help='device id (i.e. 0 or 0,1) or cpu')
        parser.add_argument('--view-img', action='store_true', help='display results')
        parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
        parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
        parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')

        opt = parser.parse_args('')

        yolov3 = yolov3_ultralytics(opt=opt)
        # yolov3 = detect(opt=opt)
        return yolov3


