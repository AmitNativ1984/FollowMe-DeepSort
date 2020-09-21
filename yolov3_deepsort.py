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

        self.cfg.CLS_DICT.pop('merge_from_dict')
        self.cfg.CLS_DICT.pop('merge_from_file')
        self.cls_dict = dict(zip([int(k) for k in list(cfg.CLS_DICT.keys())], cfg.CLS_DICT.values()))

        self.vdo = cv2.VideoCapture()

        # Define dataloader
        kwargs = {'batch_size': args.batch_size, 'pin_memory': True}
        dataset= ProbotSensorsDataset(args)
        self.data_loader = torch.utils.data.DataLoader(dataset, **kwargs, shuffle=False)

        self.cam2world = Cam2World(args.img_width, args.img_height,
                              args.thetaX, args.target_height, camAngle=0, cam_pos=cfg.CAMERA2IMU_DIST)

        self.detector = build_detector(cfg, args, use_cuda=use_cuda)
        self.deepsort = build_tracker(cfg, use_cuda=use_cuda)
        self.class_names = self.detector.class_names
        self.bbox_xyxy = []
        self.identities = []
        self.target_id = []


    def run(self):
        VISUALIZE_DETECTIONS = True
        plt.ion()
        new_traget_raw = True
        U = np.array([[0], [0], [0], [0]])


        fig3Dtracking = plt.figure(constrained_layout=True, figsize=(6*3,3*3))
        figGridSpec = fig3Dtracking.add_gridspec(nrows=3, ncols=2, left=0.05, right=0.48, wspace=0.05)
        figImage = fig3Dtracking.add_subplot(figGridSpec[:2, :])
        figMap = fig3Dtracking.add_subplot(figGridSpec[-1, 0])
        figMap.set_aspect('equal', 'box')
        figMap.grid(True)
        figRadar = fig3Dtracking.add_subplot(figGridSpec[-1, -1], projection='polar')
        figRadar.set_aspect('equal', 'box')

        figRadar.set_theta_direction(-1)
        figRadar.set_theta_zero_location("N")
        figRadar.set_rlabel_position(90)
        figRadar.set_rlim(bottom=0, top=30)

        # figRadar.set_ylim(0, max(ylim))

        plt.tight_layout()

        # radar_fig, ax_polar, ax_carthesian = create_radar_plot(radar_fig=figRadar, ax_polar=figRadar, ax_carthesian=figRadar)

        for frame, sample in enumerate(self.data_loader):
            start_time = time.time()

            sample["telemetry"] = dict(zip(sample["telemetry"].keys(), [v.numpy() for v in sample["telemetry"].values()]))
            # update cam2world functions with new telemetry:
            self.cam2world.digest_new_telemetry(sample["telemetry"])
            ori_im = sample["image"].numpy().squeeze(0)

            tracks, detections = self.DeepSort(sample, self.cam2world)
            tracking_time = time.time()

            # draw boxes for visualization
            if args.display:
                if frame == 0:
                    image_ax = figImage.imshow(ori_im)

                ori_im = draw_boxes(ori_im,
                                    [detection.to_tlbr() for detection in detections],
                                    confidence=[detection.confidence for detection in detections],
                                    track_id=[""] * np.shape(detections)[0],
                                    target_xyz=[detection.to_utm() for detection in detections],
                                    cls_names=[self.cls_dict[int(detection.cls_id)] for detection in detections],
                                    color=[255, 255, 0])
                # ax_map.scatter(target_utm_temp[0], target_utm_temp[1], marker='^', color='g', s=15)

            if self.args.display:
                figMap.scatter(sample["telemetry"]["utmpos"][0], sample["telemetry"]["utmpos"][1],
                               marker='o', color='b', alpha=0.5)
                if len(tracks) > 0:
                    confidence = [track.confidence for track in tracks]
                    track_id = [track.track_id for track in tracks]
                    cls_names = [self.cls_dict[int(track.cls_id)] for track in tracks]
                    utm_pos = [track.utm_pos for track in tracks]

                    bbox_tlwh = np.array([track.bbox_tlwh for track in tracks])
                    bbox_tlbr = bbox_tlwh
                    bbox_tlbr[:, 2] = bbox_tlwh[:, 0] + bbox_tlwh[:, 2]
                    bbox_tlbr[:, 3] = bbox_tlwh[:, 1] + bbox_tlwh[:, 3]

                    ori_im = draw_boxes(ori_im,
                                        bbox_tlbr,
                                        confidence=confidence,
                                        track_id=track_id,
                                        target_xyz=utm_pos,
                                        cls_names=cls_names,
                                        color=[255, 0, 0])

                    figMap.scatter(utm_pos[0][0], utm_pos[0][1], marker='^',
                                   color='r', alpha=0.5)

                    figRadar.cla()
                    figRadar.set_theta_direction(-1)
                    figRadar.set_rlabel_position(90)
                    figRadar.set_rlim(bottom=0, top=20)
                    figRadar.set_theta_zero_location("N")#, offset=+np.rad2deg(sample["telemetry"]["yaw_pitch_roll"][0][0]))
                    relxyz = self.cam2world.convert_bbox_tlbr_to_relative_to_camera_xyz(bbox_tlbr[0])
                    Rz = relxyz[1]
                    Rx = relxyz[0]
                    R = np.sqrt(Rz ** 2 + Rx ** 2)
                    theta_deg = np.arctan2(Rx, Rz)
                    figRadar.scatter(theta_deg+sample["telemetry"]["yaw_pitch_roll"][0][0], R)

                    figRadar.set_theta_zero_location(
                        "N")#, offset=+np.rad2deg(sample["telemetry"]["yaw_pitch_roll"][0][0]))
                    #
                    # figRadar.set_thetagrids(np.linspace(-self.args.thetaX/2, self.args.thetaX/2, num=5) +
                    #                         np.rad2deg(sample["telemetry"]["yaw_pitch_roll"][0][0]))

                    # figRadar.set_thetamin(int(-self.args.thetaX/2 + np.rad2deg(sample["telemetry"]["yaw_pitch_roll"][0][0])))
                    # figRadar.set_thetamax(int(+self.args.thetaX/2 + np.rad2deg(sample["telemetry"]["yaw_pitch_roll"][0][0])))



                image_ax.set_data(ori_im)


                end_time = time.time()
                print('frame: {}, total time: {:.3f}[msec], tracking time: {:.3f}[msec], visualize time: {:.3f}[msec]'.format(frame, (end_time - start_time) * 1E3, (tracking_time-start_time) * 1E3, (end_time - tracking_time) * 1E3))
                plt.pause(0.000001)
            if self.args.save_path:
                self.writer.write(ori_im)

            plt.ioff()


    def DeepSort(self, sample, camera2world):

        im = sample["image"].numpy().squeeze(0)
        tracks = []
        detections = []

        # do detection
        bbox_xywh, cls_conf, cls_ids = self.detector(im)  # get all detections from image

        # if bbox_xywh[0] is not None:
        if bbox_xywh[0].shape[0] > 0:
            mask = np.isin(cls_ids[0], list(self.cls_dict.keys()))
            bbox_xywh = bbox_xywh[0][mask]
            cls_conf = cls_conf[0][mask]
            cls_ids = cls_ids[0][mask]

            # run deepsort algorithm to match detections to tracks
            tracks, detections = self.deepsort.update(camera2world, bbox_xywh, cls_conf, cls_ids, im)

        return tracks, detections

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
