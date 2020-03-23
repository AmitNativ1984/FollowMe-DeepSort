import os
import cv2
import time
import argparse
import torch
import numpy as np
from distutils.util import strtobool

from detector import build_detector
from deep_sort import build_tracker, build_fuser
from deep_sort.muticam_fuser import DeepSortFuser
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
            cv2.namedWindow("cam0", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("cam0", args.display_width, args.display_height)
            cv2.moveWindow("cam0", 100 + args.display_width, 0)
            cv2.namedWindow("cam1", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("cam1", args.display_width, args.display_height)
            cv2.moveWindow("cam1", 100, 0)
            cv2.namedWindow("cam2", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("cam2", args.display_width, args.display_height)
            cv2.moveWindow("cam2", 100 + 2 * args.display_width, 0)

        self.cam2world = Cam2World(self.args.img_width, self.args.img_height,
                                   self.args.thetaX)
        self.cls_dict = {0: 'person', 2: 'car', 7: 'car'}
        # self.cam0 = cv2.VideoCapture()
        self.detector = build_detector(cfg, args, use_cuda=use_cuda)
        self.deepsort_cam0 = build_tracker(cfg, use_cuda=use_cuda)
        self.deepsort_cam1 = build_tracker(cfg, use_cuda=use_cuda)
        self.deepsort_cam2 = build_tracker(cfg, use_cuda=use_cuda)

        self.cam_fuser = build_fuser(cfg, use_cuda=use_cuda)

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
        assert os.path.isfile(self.args.cam0), "Error: path error"
        self.active_cams = []
        # self.im_width = int(self.cam0.get(cv2.CAP_PROP_FRAME_WIDTH))
        # self.im_height = int(self.cam0.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.active_cams.append(self.args.cam0)
        # opening other streams if they exist
        if self.args.cam1 != None:
            assert os.path.isfile(self.args.cam1), "Error: path error"
            self.active_cams.append(self.args.cam1)

        if self.args.cam2 != None:
            assert os.path.isfile(self.args.cam2), "Error: path error"
            self.active_cams.append(self.args.cam2)

        # if self.args.save_path:
        #     fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        #     self.writer = cv2.VideoWriter(self.args.save_path, fourcc, 20, (self.im_width, self.im_height))

        # assert self.cam0.isOpened()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)


    def run(self):
        frame = 0
        target_cls = list(self.cls_dict.keys())
        target_name = list(self.cls_dict.values())

        cap = [cv2.VideoCapture(i) for i in self.active_cams]
        num_cameras = len(cap)
        while True:
            start = time.time()
            img_batch= []
            for i, c in enumerate(cap):
                if c is not None:
                    ret, read_img = c.read()
                    if ret == True:
                        read_img = cv2.cvtColor(read_img, cv2.COLOR_BGR2RGB)
                        if img_batch == []:
                            img_batch = np.array(read_img)
                        else:
                            img_batch = np.concatenate((img_batch, read_img), axis=-1)
                    if ret == False:
                        break
            if ret == False:
                break

            im = img_batch
            # todo: convert multiimage to batch in pytorch
            print('frame: {}'.format(frame))
            # do detection on all camera streams simultaneously
            bbox_xywh, cls_conf, cls_ids = self.detector(im)  # get all detections from image

            # running DeepSort on every camera separately
            if bbox_xywh != []:
                cam_data = []
                for cam_id in range(num_cameras):

                    im[..., 3 * cam_id:3 * (cam_id + 1)] = cv2.cvtColor(im[..., 3*cam_id:3*(cam_id+1)], cv2.COLOR_BGR2RGB)
                    img = im[..., 3*cam_id:3*(cam_id+1)].copy()
                    outputs, detections, detections_conf, target_xyz, cls_names = self.DeepSort(img, bbox_xywh[cam_id], cls_conf[cam_id], cls_ids[cam_id], target_cls, cam_id=cam_id)

                    data = {
                            "outputs": outputs,
                            "detections": detections,
                            "detection_conf": detections_conf,
                            "target_xyz": target_xyz,
                            "cls_names": cls_names
                            }
                    cam_data.append(data)

                    # draw boxes for visualization
                    if len(detections) > 0:
                        DetectionsBBOXES = np.array([detections[i].tlwh for i in range(np.shape(detections)[0])])
                        DetectionsBBOXES[:, 2] += DetectionsBBOXES[:, 0]
                        DetectionsBBOXES[:, 3] += DetectionsBBOXES[:, 1]
                        bbox_xyxy = DetectionsBBOXES[:, :4]
                        identities = np.zeros(DetectionsBBOXES.shape[0])
                        img = draw_boxes(img, bbox_xyxy, identities, target_id=self.target_id,
                                            target_xyz=target_xyz, cls_names=cls_names, clr=[0, 255, 0])

                    if len(outputs) > 0:
                        bbox_xyxy = outputs[:, :4]
                        identities = outputs[:, -1]
                        img = draw_boxes(img, bbox_xyxy, identities, target_id=self.target_id, confs=detections_conf,
                                            target_xyz=target_xyz, cls_names=cls_names)
                        self.bbox_xyxy = bbox_xyxy
                        self.identities = identities
                    if self.args.display:
                        cv2.imshow("cam"+str(cam_id),img)
                        key = cv2.waitKey(1)
                        # cv2.setMouseCallback("cam"+str(b), self.select_target)

                    if self.args.save_path:
                        self.writer.write(img)

                    frame += 1

                    for i, target in enumerate(outputs):
                        id = target[-1]
                        print(
                            '\t\t target [{}] \t ({},{},{},{})\t conf {:.2f}\t position: ({:.2f}, {:.2f}, {:.2f})[m]\t cls: [{}]'
                                .format(id, target[0], target[1], target[2], target[3], detections_conf[i],
                                        target_xyz[i][0], target_xyz[i][1], target_xyz[i][2], cls_names[i]))

                # the purpose of this step is to give same id to an object tracked in different cameras
                # outputs, detections, detections_conf, target_xyz, cls_names = self.FuseCams(cam_data)
                # detections = self.cam_fuser.fuse(cam_data)

    def DeepSort(self, im, bbox_xywh, cls_conf, cls_ids, target_cls, cam_id=0):

        outputs = []
        detections = []
        detections_conf = []
        cls_id = []
        target_xyz = []


        if bbox_xywh is not None:
            # select person class
            for cls in target_cls:
                try:
                    mask += cls_ids == cls
                except Exception:
                    mask = cls_ids == cls

            bbox_xywh = bbox_xywh[mask]
            cls_conf = cls_conf[mask]
            cls_ids = cls_ids[mask]

            # run deepsort algorithm to match detections to tracks
            if cam_id == 0:
                outputs, detections, detections_conf, cls_id = self.deepsort_cam0.update(bbox_xywh, cls_conf, cls_ids, im)
            elif cam_id == 1:
                outputs, detections, detections_conf, cls_id = self.deepsort_cam1.update(bbox_xywh, cls_conf, cls_ids, im)
            elif cam_id == 2:
                outputs, detections, detections_conf, cls_id = self.deepsort_cam2.update(bbox_xywh, cls_conf, cls_ids, im)
            # calculate object distance and direction from camera
            target_xyz = []
            for i, target in enumerate(outputs):
                target_xyz.append(self.cam2world.obj_world_xyz(x1=target[0],
                                                               y1=target[1],
                                                               x2=target[2],
                                                               y2=target[3],
                                                               obj_height_meters=self.args.target_height))

        cls_name = [list(self.cls_dict.values())[int(target_cls)] for target_cls in cls_id]
        return outputs, detections, detections_conf, target_xyz, cls_name


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("VIDEO_PATH", type=str)
    parser.add_argument("--cam0", type=str, required=True,
                        help="center cam stream")
    parser.add_argument("--cam1", type=str, required=False,
                        help="left cam stream")
    parser.add_argument("--cam2", type=str, required=False,
                        help="right cam stream")
    parser.add_argument("--config_detection", type=str, default="./configs/yolov3_probot.yaml")
    parser.add_argument("--config_deepsort", type=str, default="./configs/deep_sort.yaml")
    parser.add_argument("--ignore_display", dest="display", action="store_false", default=True)
    parser.add_argument("--display_width", type=int, default=800)
    parser.add_argument("--display_height", type=int, default=600)
    parser.add_argument("--save_path", type=str, default="")
    parser.add_argument("--cpu", dest="use_cuda", action="store_false", default=True)
    parser.add_argument("--target_cls", type=str, default='0', help='coco dataset labels to track')
    parser.add_argument("--yolo-method", type=str, default='ultralytics', choices=['ultralytics', 'org'],
                        help='yolo backbone method. can be one of: [ultralytics, org]')
    parser.add_argument("--img-width", type=int, default=1280,
                        help='img width in pixels')
    parser.add_argument("--img-height", type=int, default=720,
                        help='img height in pixels')
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

    with Tracker(cfg, args) as tracker:
        tracker.run()
