import os
import cv2
import time
import argparse
import torch
import numpy as np
from distutils.util import strtobool
import matplotlib.pyplot as plt
from deep_sort.dataloaders.vehicle_sensors_dataloader import ProbotSensorsDataset
from detector import build_detector
from deep_sort import build_tracker
from utils.draw import draw_boxes, create_radar_plot
from utils.parser import get_config
from utils.camera2world import Cam2World
from deep_sort.sort.kalman_filter_xyz import KalmanXYZ

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

        self.cfg.CLS_DICT.pop('merge_from_dict')
        self.cfg.CLS_DICT.pop('merge_from_file')
        self.cls_dict = dict(zip([int(k) for k in list(cfg.CLS_DICT.keys())], cfg.CLS_DICT.values()))

        self.vdo = cv2.VideoCapture()

        # Define dataloader
        kwargs = {'batch_size': args.batch_size, 'pin_memory': True}
        dataset= ProbotSensorsDataset(args)
        self.data_loader = torch.utils.data.DataLoader(dataset, **kwargs, shuffle=False)
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

    def run(self):
        VISUALIZE_DETECTIONS = True
        target_cls = list(self.cls_dict.keys())
        target_name = list(self.cls_dict.values())
        cam2world = Cam2World(self.args.img_width, self.args.img_height,
                              self.args.thetaX, args.target_height, camAngle=0, cam_pos=cfg.CAMERA2IMU_DIST)

        new_traget=True
        new_traget_raw = True
        U = np.array([[0], [0], [0], [0]])
        map_projection_fig = plt.figure()
        ax_map = map_projection_fig.add_subplot(111)
        ax_map.set_aspect('equal', 'box')
        ax_map.grid(True)
        radar_fig, ax_polar, ax_carthesian = create_radar_plot()

        for frame, sample in enumerate(self.data_loader):
            start = time.time()
            print('frame: {}'.format(frame))

            sample["telemetry"] = dict(zip(sample["telemetry"].keys(), [v.numpy() for v in sample["telemetry"].values()]))
            # update cam2world functions with new telemetry:
            cam2world.process_new_telemetry(sample["telemetry"])
            ori_im = sample["image"].numpy().squeeze(0)

            outputs, detections, detections_conf, target_xyz, cls_names = self.DeepSort(ori_im, target_cls)

            # draw boxes for visualization


            # todo: this if just for "quick and dirt" version of Kalman filter"
            # kalman filter predictions on detection [0]
            for detection in detections:
                target_utm_temp = cam2world.convert_bbox_tlbr_to_utm_coordinates(detection.to_tlbr())
                # drawing...
                if VISUALIZE_DETECTIONS:
                    ori_im = draw_boxes(ori_im, [detections[0].to_tlbr()], confidence=[detections[0].confidence],
                                        track_id=[""],
                                        target_xyz=[target_utm_temp],
                                        cls_names=[self.cls_dict[int(detections[0].cls_id)]],
                                        color=[255, 255, 0])
                    ax_map.scatter(target_utm_temp[0], target_utm_temp[1], marker='^', color='g', s=15)

                if new_traget_raw:
                    kalman_xyz_raw = KalmanXYZ(timestamp=sample["telemetry"]["timestamp"][0],
                                           x0=target_utm_temp)
                    new_traget_raw = False

                kalman_xyz_raw.predict(sample["telemetry"]["timestamp"][0], U)
                x_statenew_raw = kalman_xyz_raw.update_by_measurment(target_utm_temp)
                ax_map.scatter(np.array(x_statenew_raw)[0], np.array(x_statenew_raw)[1], marker='^', color='black', s=15, alpha=0.5)
                target_utm_new = np.vstack((x_statenew_raw[:2], target_utm_temp[-1]))
                # converting back to bbox from utm
                bbox_row_center, bbox_col_center = cam2world.convert_utm_coordinates_to_bbox_center(target_utm_new)

                bbox_xyxy_temp = np.array([detections[0].to_tlbr() for i in range(np.shape(detections)[0])])
                bbox_xyxy_temp[:, 0::2] += bbox_col_center - np.mean([bbox_xyxy_temp[:, 0], bbox_xyxy_temp[:, 2]])
                bbox_xyxy_temp[:, 1::2] += bbox_row_center - np.mean([bbox_xyxy_temp[:, 1], bbox_xyxy_temp[:, 3]])

                ori_im = draw_boxes(ori_im, bbox_xyxy_temp, track_id=[""], confidence= [detections[0].confidence],
                                    target_xyz=[target_utm_temp], cls_names=[self.cls_dict[int(detections[0].cls_id)]],
                                    color=[0, 50, 255])
                break

            if len(outputs) > 0:
                bbox_xyxy = outputs[:, :4]
                identities = outputs[:, -1]
                ori_im = draw_boxes(ori_im, bbox_xyxy, track_id=identities, confidence=detections_conf,
                                    target_xyz=target_xyz, cls_names=cls_names, color=[255, 0, 0])
                self.bbox_xyxy = bbox_xyxy
                self.identities = identities

                for i, target in enumerate(outputs):
                    id = target[-1]
                    print('\t\t target [{}] \t cls: [{}]\t ({},{},{},{})\t conf {:.2f}\t'
                          .format(id, cls_names[0], target[0], target[1], target[2], target[3],
                                  detections_conf[i]))  # ,
                                    #target_xyz[i][0], target_xyz[i][1], target_xyz[i][2], cls_names[i]))

            if self.args.display:
                cv2.imshow("test", cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB))
                key = cv2.waitKey(1)
                cv2.setMouseCallback("test", self.select_target)
                radar_fig, ax_polar, ax_carthesian = create_radar_plot(radar_fig=radar_fig, ax_polar=ax_polar,
                                                                       ax_carthesian=ax_carthesian)
                ax_map.scatter(sample["telemetry"]["utmpos"][0], sample["telemetry"]["utmpos"][1], marker='o', color='b', alpha=0.5)






            plt.pause(0.1)


            if self.args.save_path:
                self.writer.write(ori_im)



    def DeepSort(self, im, target_cls):

        # do detection
        bbox_xywh, cls_conf, cls_ids = self.detector(im)    # get all detections from image
        outputs = []
        detections = []
        detections_conf = []
        cls_name = []
        target_xyz = []

        if bbox_xywh[0] is not None:
            # select person class
            mask = np.isin(cls_ids[0], list(self.cls_dict.keys()))

            bbox_xywh = bbox_xywh[0][mask]
            cls_conf = cls_conf[0][mask]
            cls_ids = cls_ids[0][mask]

            # run deepsort algorithm to match detections to tracks
            outputs, detections, detections_conf, cls_ids = self.deepsort.update(bbox_xywh, cls_conf, cls_ids, im)

            # calculate object distance and direction from camera
            target_xyz = []
            for i, target in enumerate(outputs):
                bbox_tlbr = target[:4]
                # target_xyz.append(self.cam2world.obj_xyz_relative_to_camera(bbox_tlbr,
                #                                                             obj_height_meters=self.args.target_height))
                cls_name = [self.cls_dict[int(id)] for id in cls_ids]

        return outputs, detections, detections_conf, target_xyz, cls_name

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help='folder containing all recorded data')
    parser.add_argument("--batch-size", type=int, default=1, help='folder containing all recorded data')
    parser.add_argument("--config_detection", type=str, default="./configs/yolov3_probot_ultralytics.yaml")
    parser.add_argument("--config_deepsort", type=str, default="./configs/deep_sort.yaml")
    parser.add_argument("--config_sensors", type=str, default="./configs/sensors.yaml")
    parser.add_argument("--ignore_display", dest="display", action="store_false", default=True)
    parser.add_argument("--display_width", type=int, default=800)
    parser.add_argument("--display_height", type=int, default=600)
    parser.add_argument("--save_path", type=str, default="")#"./demo/demo.avi")
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
    parser.add_argument("--target-height", type=float, default=1.7,
                        help='tracked target height in [meters]')

    return parser.parse_args()
# 90x 67.5y

if __name__=="__main__":
    args = parse_args()
    cfg = get_config()
    cfg.merge_from_file(args.config_detection)
    cfg.merge_from_file(args.config_deepsort)
    cfg.merge_from_file(args.config_sensors)

    torch.device("cuda:1")

    Tracker(cfg, args).run()

    plt.show()
