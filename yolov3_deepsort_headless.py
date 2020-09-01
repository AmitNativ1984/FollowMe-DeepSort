import os
from os.path import dirname, abspath, join
import time
import argparse
import torch
import numpy as np
import sys
import ctypes
from distutils.util import strtobool

from detector import build_detector
from deep_sort import build_tracker
from utils.parser import get_config
from utils.camera2world import Cam2World

def get_merged_config():
    cfg = get_config()

    cfg.merge_from_file(join(dirname(__file__), 'configs', 'yolov3_probot_ultralytics.yaml'))
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

        self.cam2world = Cam2World(self.width, self.height, self.camera_fov_x)
        self.cls_dict = {0: 'person', 2: 'car', 7: 'car'}

        args = lambda: None
        args.yolo_method = 'org'
        self.detector = build_detector(self.cfg, args, use_cuda=use_cuda)
        self.deepsort = build_tracker(self.cfg, use_cuda=use_cuda)
        self.class_names = self.detector.class_names
        self.bbox_xyxy = []
        self.identities = []
        self.target_id = []

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
