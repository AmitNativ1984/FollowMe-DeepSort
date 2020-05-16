import clr, System
import os
import cv2
import time
import argparse
import torch
import numpy as np
import sys
import ctypes
from distutils.util import strtobool
from System import Array, Int32
from System.Runtime.InteropServices import GCHandle, GCHandleType

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
                                   self.args.thetaX)
        self.cls_dict = {0: 'person', 2: 'car', 7: 'car'}
        self.vdo = cv2.VideoCapture()
        self.detector = build_detector(cfg, args, use_cuda=use_cuda)
        self.deepsort = build_tracker(cfg, use_cuda=use_cuda)
        self.class_names = self.detector.class_names
        self.bbox_xyxy = []
        self.identities = []
        self.target_id = []


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
            fourcc =  cv2.VideoWriter_fourcc(*'MP4V')
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

            print('frame: {}'.format(frame))
            outputs, detections, detections_conf, target_xyz, cls_names = self.DeepSort(im, target_cls)


            # draw boxes for visualization
            if len(detections) > 0:
                DetectionsBBOXES = np.array([detections[i].tlwh for i in range(np.shape(detections)[0])])
                DetectionsBBOXES[:, 2] += DetectionsBBOXES[:, 0]
                DetectionsBBOXES[:, 3] += DetectionsBBOXES[:, 1]
                bbox_xyxy = DetectionsBBOXES[:, :4]
                identities = np.zeros(DetectionsBBOXES.shape[0])
                ori_im = draw_boxes(ori_im, bbox_xyxy, identities, target_id=self.target_id,
                                    target_xyz=target_xyz, cls_names=cls_names, clr=[0, 255, 0])

            if len(outputs) > 0:
                bbox_xyxy = outputs[:, :4]
                identities = outputs[:, -1]
                ori_im = draw_boxes(ori_im, bbox_xyxy, identities, target_id=self.target_id, confs=detections_conf,
                                    target_xyz=target_xyz, cls_names=cls_names)
                self.bbox_xyxy = bbox_xyxy
                self.identities = identities
            if self.args.display:
                cv2.imshow("test", ori_im)
                key = cv2.waitKey(1)
                cv2.setMouseCallback("test", self.select_target)

            if self.args.save_path:
                self.writer.write(ori_im)

            frame += 1


        if self.args.save_path:
            self.writer.release()

    def DeepSort(self, im, target_cls):
        im = self.asNumpyArray(im)
        im = im.reshape(720, 1280, 3)

        # do detection
        bbox_xywh, cls_conf, cls_ids = self.detector(im)    # get all detections from image
        outputs = []
        detections = []
        detections_conf = []
        cls_id = []
        target_xyz = []

        if bbox_xywh is not None:
            # select person class
            mask = np.isin(cls_ids[0], list(self.cls_dict.keys()))

            bbox_xywh = bbox_xywh[0][mask]
            cls_conf = cls_conf[0][mask]
            cls_ids = cls_ids[0][mask]

            # run deepsort algorithm to match detections to tracks
            outputs, detections, detections_conf, cls_id = self.deepsort.update(bbox_xywh, cls_conf, cls_ids, im)

            # calculate object distance and direction from camera
            target_xyz = []
            for i, target in enumerate(outputs):
                target_xyz.append(self.cam2world.obj_world_xyz(x1=target[0],
                                                               y1=target[1],
                                                               x2=target[2],
                                                               y2=target[3],
                                                               obj_height_meters=self.args.target_height))

        cls_name = [self.cls_dict[int(target_cls)] for target_cls in cls_id]

        return outputs, detections, detections_conf, target_xyz, cls_name, len(outputs)


    def asNumpyArray(self, netArray):
        '''
        Given a CLR `System.Array` returns a `numpy.ndarray`.  See _MAP_NET_NP for
        the mapping of CLR types to Numpy dtypes.
        '''

        _MAP_NET_NP = {
        'Single' : np.dtype('float32'),
        'Double' : np.dtype('float64'),
        'SByte'  : np.dtype('int8'),
        'Int16'  : np.dtype('int16'),
        'Int32'  : np.dtype('int32'),
        'Int64'  : np.dtype('int64'),
        'Byte'   : np.dtype('uint8'),
        'UInt16' : np.dtype('uint16'),
        'UInt32' : np.dtype('uint32'),
        'UInt64' : np.dtype('uint64'),
        'Boolean': np.dtype('bool'),
    }


        dims = np.empty(netArray.Rank, dtype=int)
        for I in range(netArray.Rank):
            dims[I] = netArray.GetLength(I)
        netType = netArray.GetType().GetElementType().Name

        try:
            npArray = np.empty(dims, order='C', dtype=_MAP_NET_NP[netType])
        except KeyError:
            raise NotImplementedError("asNumpyArray does not yet support System type {}".format(netType) )

        try: # Memmove
            sourceHandle = GCHandle.Alloc(netArray, GCHandleType.Pinned)
            sourcePtr = sourceHandle.AddrOfPinnedObject().ToInt64()
            destPtr = npArray.__array_interface__['data'][0]
            ctypes.memmove(destPtr, sourcePtr, npArray.nbytes)
        finally:
            if sourceHandle.IsAllocated: sourceHandle.Free()
        return npArray

def parse_args():
    parser = argparse.ArgumentParser()
   # parser.add_argument("VIDEO_PATH", type=str)
    parser.add_argument("--config_detection", type=str, default="./configs/yolov3.yaml")
    parser.add_argument("--config_deepsort", type=str, default="./configs/deep_sort.yaml")
    parser.add_argument("--ignore_display", dest="display", action="store_false", default=False)
    parser.add_argument("--display_width", type=int, default=800)
    parser.add_argument("--display_height", type=int, default=600)
    parser.add_argument("--save_path", type=str, default="./demo/demo.avi")
    parser.add_argument("--cpu", dest="use_cuda", action="store_false", default=True)
    parser.add_argument("--target_cls", type=str, default='0', help='coco dataset labels to track')
    parser.add_argument("--yolo-method", type=str, default='org', choices=['ultralytics', 'org'],
                        help='yolo backbone method. can be one of: [ultralytics, org]')
    parser.add_argument("--img-width", type=int, default=1280,
                        help='img width in pixels')
    parser.add_argument("--img-height", type=int, default=720,
                        help='img height in pixels')
    parser.add_argument("--thetaX", type=float, default=77.04,
                        help='angular camera FOV in horizontal direction. [deg]')
    parser.add_argument("--target-height", type=float, default=1.8,
                        help='tracked target height in [meters]')

    return parser.parse_args()


if __name__=="__main__":
    args = parse_args()
    cfg = get_config()
    cfg.merge_from_file(args.config_detection)
    cfg.merge_from_file(args.config_deepsort)

    torch.device("cuda:1")

    with Tracker(cfg, args) as tracker:
        tracker.run()
