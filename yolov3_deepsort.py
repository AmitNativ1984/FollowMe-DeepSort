import os
import cv2
import time
import argparse
import torch
import numpy as np
from distutils.util import strtobool
import matplotlib.pyplot as plt

from detector import build_detector
from deep_sort import build_tracker
from utils.draw import draw_boxes, create_radar_plot
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
            fourcc =  cv2.VideoWriter_fourcc(*'MJPG')
            self.writer = cv2.VideoWriter(self.args.save_path, fourcc, 10, (self.im_width, self.im_height))

        assert self.vdo.isOpened()
        return self

    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)
        

    def run(self):
        VISUALIZE_DETECTIONS = True
        frame = 0
        target_cls = list(self.cls_dict.keys())
        target_name = list(self.cls_dict.values())

        # radar_fig, ax_polar, ax_carthesian = create_radar_plot()

        while self.vdo.grab():
            start = time.time()
            _, ori_im = self.vdo.retrieve()
            im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)

            print('frame: {}'.format(frame))
            outputs, detections, detections_conf, target_xyz, cls_names = self.DeepSort(im, target_cls)


            # draw boxes for visualization
            if len(detections) > 0 and VISUALIZE_DETECTIONS:
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

            frame += 1

            if self.args.display:
                cv2.imshow("test", ori_im)
                key = cv2.waitKey(1)
                cv2.setMouseCallback("test", self.select_target)
                # radar_fig, ax_polar, ax_carthesian = create_radar_plot(radar_fig=radar_fig, ax_polar=ax_polar,
                #                                                        ax_carthesian=ax_carthesian)

            for i, target in enumerate(outputs):

                id = target[-1]
                print(
                    '\t\t target [{}] \t ({},{},{},{})\t conf {:.2f}\t position: ({:.2f}, {:.2f}, {:.2f})[m]\t cls: [{}]'
                    .format(id, target[0], target[1], target[2], target[3], detections_conf[i],
                            target_xyz[i][0], target_xyz[i][1], target_xyz[i][2], cls_names[i]))

                # ax_carthesian.scatter(target_xyz[i][0], target_xyz[i][2], marker='x', color='r')
                # ax_carthesian.scatter(1, 3, marker='x', color='b')

            plt.pause(0.1)


            if self.args.save_path:
                self.writer.write(ori_im)

        self.writer.release()
        cv2.destroyAllWindows()

    def DeepSort(self, im, target_cls):
        # do detection
        bbox_xywh, cls_conf, cls_ids = self.detector(im)    # get all detections from image
        bbox_xywh = bbox_xywh[0]
        cls_conf = cls_conf[0]
        cls_ids = cls_ids[0]
        outputs = []
        detections = []
        detections_conf = []
        cls_name = []
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
            outputs, detections, detections_conf, cls_ids = self.deepsort.update(bbox_xywh, cls_conf, cls_ids, im)

            # calculate object distance and direction from camera
            target_xyz = []
            for i, target in enumerate(outputs):
                target_xyz.append(self.cam2world.obj_world_xyz(x1=target[0],
                                                               y1=target[1],
                                                               x2=target[2],
                                                               y2=target[3],
                                                               obj_height_meters=self.args.target_height))
                cls_name = [self.cls_dict[int(id)] for id in cls_ids]

        return outputs, detections, detections_conf, target_xyz, cls_name

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("VIDEO_PATH", type=str)
    parser.add_argument("--config_detection", type=str, default="./configs/yolov3_probot.yaml")
    parser.add_argument("--config_deepsort", type=str, default="./configs/deep_sort.yaml")
    parser.add_argument("--ignore_display", dest="display", action="store_false", default=True)
    parser.add_argument("--display_width", type=int, default=800)
    parser.add_argument("--display_height", type=int, default=600)
    parser.add_argument("--save_path", type=str, default="./demo/woods_camouflage_test.avi")
    parser.add_argument("--cpu", dest="use_cuda", action="store_false", default=True)
    parser.add_argument("--target_cls", type=str, default='0', help='coco dataset labels to track')
    parser.add_argument("--yolo-method", type=str, default='ultralytics', choices=['ultralytics', 'org'],
                        help='yolo backbone method. can be one of: [ultralytics, org]')
    parser.add_argument("--img-width", type=int, default=1280,
                        help='img width in pixels')
    parser.add_argument("--img-height", type=int, default=720,
                        help='img height in pixels')
    parser.add_argument("--thetaX", type=float, default=77.04,#62.0,
                        help='angular camera FOV in horizontal direction. [deg]')
    parser.add_argument("--target-height", type=float, default=1.8,
                        help='tracked target height in [meters]')

    return parser.parse_args()
# 90x 67.5y

if __name__=="__main__":
    args = parse_args()
    cfg = get_config()
    cfg.merge_from_file(args.config_detection)
    cfg.merge_from_file(args.config_deepsort)

    torch.device("cuda:1")

    with Tracker(cfg, args) as tracker:
        tracker.run()

    plt.show()
