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

        self.cam2world = Cam2World(self.args.img_width, self.args.img_height,
                                   self.args.thetaX)
        self.cls_dict = {0: 'person', 2: 'car', 7: 'car'}
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
            # _, ori_im = self.vdo.retrieve()

            ori_im = sample["image"].numpy().squeeze(0)
            ori_im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)
            robot_xyz = sample["telemetry"]["utmpos"][0]
            robot_yaw = sample["telemetry"]["yaw_pitch_roll"][0][0].numpy() * np.pi / 180

            rotation_matrix = np.array([[np.cos(robot_yaw), 0, np.sin(robot_yaw)],
                                        [0, 0, 0],
                                        [-np.sin(robot_yaw), 0, np.cos(robot_yaw)]])

            camera_imu_dist = 1.7

            im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)

            print('frame: {}'.format(frame))
            outputs, detections, detections_conf, target_xyz, cls_names = self.DeepSort(im, target_cls)

            # todo: SHOULD IT BE TARGET XYZ BEFORE/AFTER KALMAN FILTER????

            # draw boxes for visualization
            if len(detections) > 0 and VISUALIZE_DETECTIONS:
                DetectionsBBOXES = np.array([detections[i].tlwh for i in range(np.shape(detections)[0])])
                DetectionsBBOXES[:, 2] += DetectionsBBOXES[:, 0]
                DetectionsBBOXES[:, 3] += DetectionsBBOXES[:, 1]
                bbox_xyxy = DetectionsBBOXES[:, :4]
                identities = np.zeros(DetectionsBBOXES.shape[0])


                target_xyz_temp = self.cam2world.obj_world_xyz(x1=bbox_xyxy[0, 0],
                                                          y1=bbox_xyxy[0, 1],
                                                          x2=bbox_xyxy[0, 2],
                                                          y2=bbox_xyxy[0, 3],
                                                          obj_height_meters=self.args.target_height)

                ori_im = draw_boxes(ori_im, bbox_xyxy, identities, target_id=self.target_id,
                                    target_xyz=[target_xyz_temp], cls_names=[detections[0].cls_id], clr=[0, 255, 0])

                target_xyz_temp = np.array([target_xyz_temp])
                target_xyz_temp[0, 2] += camera_imu_dist

                target_xyz_temp = np.matmul(rotation_matrix, target_xyz_temp.transpose())


                target_xyz_temp[0, 0] += robot_xyz[0].numpy()  # + camera_imu_dist
                target_xyz_temp[2, 0] += robot_xyz[1].numpy()  # + camera_imu_dist

                ax_map.scatter(target_xyz_temp[0], target_xyz_temp[2], marker='^', color='g', s=15)

                if new_traget_raw:
                    kalman_xyz_raw = KalmanXYZ(timestamp=sample["telemetry"]["timestamp"].numpy()[0],
                                           x0=np.array([[target_xyz_temp[0, 0]], [target_xyz_temp[2, 0]]]))
                    new_traget_raw=False


                kalman_xyz_raw.predict(sample["telemetry"]["timestamp"].numpy()[0], U)
                x_statenew_raw = kalman_xyz_raw.update_by_measurment(np.array([[target_xyz_temp[0,0]], [target_xyz_temp[2,0]]]))

                ax_map.scatter(np.array(x_statenew_raw)[0], np.array(x_statenew_raw)[1], marker='^', color='black', s=15, alpha=0.5)

                #todo: project from xyz back to image plane:
                target_utm = np.array([[np.asarray(x_statenew_raw)[0,0].squeeze()], [np.asarray(x_statenew_raw)[1,0].squeeze()]])
                target_xyz_new = self.cam2world.convert_target_utm_relative_xyz(robot_utm=np.array([[robot_xyz[0]], [robot_xyz[1]]]),
                                                                     target_utm=target_utm,
                                                                     robot_yaw=np.rad2deg(robot_yaw),
                                                                     camera_angle=0)

                col = self.cam2world.xyz2rowcol(target_xyz_new)
                xyz_new = [np.array([target_xyz_new[0][0], 0, target_xyz_new[1][0]])]
                # target_xyz_new = [np.array([xyz_new[0][0], 0, xyz_new[1][0]])]
                DetectionsBBOXES_temp = np.array([detections[i].tlwh for i in range(np.shape(detections)[0])])
                DetectionsBBOXES_temp[:, 0] = col - DetectionsBBOXES_temp[:, 2] / 2
                DetectionsBBOXES_temp[:, 2] += DetectionsBBOXES_temp[:, 0]
                DetectionsBBOXES_temp[:, 3] += DetectionsBBOXES_temp[:, 1]
                bbox_xyxy_temp = DetectionsBBOXES_temp[:, :4]
                ori_im = draw_boxes(ori_im, bbox_xyxy_temp, identities, target_id=self.target_id,confs= [detections[0].confidence],
                                    target_xyz=xyz_new, cls_names=[int(detections[0].confidence)], clr=[255, 50, 0])


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
                radar_fig, ax_polar, ax_carthesian = create_radar_plot(radar_fig=radar_fig, ax_polar=ax_polar,
                                                                       ax_carthesian=ax_carthesian)
                # ax_map.scatter(robot_xyz[0], robot_xyz[1], marker='o', color='b', alpha=0.5)
            for i, target in enumerate(outputs):

                id = target[-1]
                print(
                    '\t\t target [{}] \t ({},{},{},{})\t conf {:.2f}\t position: ({:.2f}, {:.2f}, {:.2f})[m]\t cls: [{}]'
                    .format(id, target[0], target[1], target[2], target[3], detections_conf[i],
                            target_xyz[i][0], target_xyz[i][1], target_xyz[i][2], cls_names[i]))


                target_xyz[i][2] += camera_imu_dist
                ax_carthesian.scatter(target_xyz[i][0], target_xyz[i][2], marker='x', color='r')

                # transofring target relative coordinates to utm system
                t_xyz = target_xyz[i][0], target_xyz[i][2]



                target_xyz[i] = np.matmul(rotation_matrix, np.expand_dims(np.array(target_xyz[i]), axis=1))
                # target_xyz[i][0] = np.cos(robot_yaw) * t_xyz[0] + np.sin(robot_yaw) * t_xyz[1]
                # target_xyz[i][2] = -np.sin(robot_yaw) * t_xyz[0] + np.cos(robot_yaw) * t_xyz[1]

                target_xyz[i][0] += robot_xyz[0].numpy()# + camera_imu_dist
                target_xyz[i][2] += robot_xyz[1].numpy()# + camera_imu_dist
                # ax_map.scatter(target_xyz[i][0], target_xyz[i][2], marker='x', color='r', s=15)
                ax_map.set_xlim([target_xyz[i][0] - 5, target_xyz[i][0] + 5])
                ax_map.set_ylim([target_xyz[i][2] - 5, target_xyz[i][2] + 5])


                # kalman filter in XYZ plane:
                if new_traget:
                    kalman_xyz = KalmanXYZ(timestamp=sample["telemetry"]["timestamp"].numpy()[0],
                                           x0=np.array([target_xyz[i][0], target_xyz[i][2]]))
                    new_traget=False


                kalman_xyz.predict(sample["telemetry"]["timestamp"].numpy()[0], U)
                x_statenew = kalman_xyz.update_by_measurment(np.array([target_xyz[i][0], target_xyz[i][2]]))
                # ax_map.scatter(np.array(x_statenew)[0], np.array(x_statenew)[1], marker='x', color='black', s=15)


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
                target_xyz.append(self.cam2world.obj_world_xyz(x1=target[0],
                                                               y1=target[1],
                                                               x2=target[2],
                                                               y2=target[3],
                                                               obj_height_meters=self.args.target_height))
                cls_name = [self.cls_dict[int(id)] for id in cls_ids]

        return outputs, detections, detections_conf, target_xyz, cls_name

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help='folder containing all recorded data')
    parser.add_argument("--batch-size", type=int, default=1, help='folder containing all recorded data')
    parser.add_argument("--config_detection", type=str, default="./configs/yolov3_probot_ultralytics.yaml")
    parser.add_argument("--config_deepsort", type=str, default="./configs/deep_sort.yaml")
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

    torch.device("cuda:1")

    # with Tracker(cfg, args) as tracker:
    #     tracker.run()

    Tracker(cfg, args).run()

    plt.show()
