from .yolov3_ultralytics.infer_img import YOLOv3 as yolov3_ultralytics
from .yolov3_ultralytics.detect import detect
import argparse

__all__ = ['build_detector']

def build_detector(cfg, use_cuda):

    if cfg.YOLOV3.BACKBONE == 'ultralytics':
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
            if use_cuda:
                opt.use_cuda = '0'
            else:
                opt.use_cuda = 'cpu'

            yolov3 = yolov3_ultralytics(opt=opt)
            # yolov3 = detect(opt=opt)
            return yolov3


