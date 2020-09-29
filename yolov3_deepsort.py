import os
import cv2
import time
import argparse
import torch
import numpy as np
from distutils.util import strtobool
import matplotlib.pyplot as plt
import ctypes
from ctypes import POINTER, c_uint8, cast

from deep_sort.dataloaders.vehicle_sensors_dataloader import ProbotSensorsDataset
from deep_sort.deep_sort import DeepSort
from utils.draw import draw_boxes, create_radar_plot
from utils.parser import get_config
from utils.camera2world import Cam2World
# from deepsort_core import DeepSort

class Tracker(object):
    def __init__(self, cfg, args):
        self.cfg = cfg
        self.args = args
        use_cuda = args.use_cuda and torch.cuda.is_available()
        if not use_cuda:
            raise UserWarning("Running in cpu mode!")

        kwargs = {'batch_size': args.batch_size, 'pin_memory': True}
        dataset = ProbotSensorsDataset(args)
        self.data_loader = torch.utils.data.DataLoader(dataset, **kwargs, shuffle=False)
        self.cam2world = Cam2World(args.img_width, args.img_height,
                                   args.thetaX, args.target_height, camAngle=0, cam_pos=cfg.CAMERA2IMU_DIST)

        self.cls_dict = {0: 'person', 2: 'car', 7: 'car'}
        self.vdo = cv2.VideoCapture()
        self.bbox_xyxy = []
        self.target_id = []
        self.DeepSort = DeepSort(self.cfg, use_cuda=use_cuda)

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)

    def run(self):
        VISUALIZE_DETECTIONS = True
        plt.ion()
        new_traget_raw = True
        U = np.array([[0], [0], [0], [0]])


        fig3Dtracking = plt.figure()
        ax_map = fig3Dtracking.add_subplot(2, 1, 1)
        ax_map.set_aspect('equal', 'box')
        ax_map.grid(True)

        ax_radar = fig3Dtracking.add_subplot(2, 1, 2, projection='polar')
        ax_radar.set_aspect('equal', 'box')
        ax_radar.set_theta_direction(-1)
        ax_radar.set_theta_zero_location("N")
        ax_radar.set_rlabel_position(90)
        ax_radar.set_rlim(bottom=0, top=30)

        plt.tight_layout()

        # radar_fig, ax_polar, ax_carthesian = create_radar_plot(radar_fig=figRadar, ax_polar=figRadar, ax_carthesian=figRadar)

        for frame, sample in enumerate(self.data_loader):
            start_time = time.time()

            sample["telemetry"] = dict(zip(sample["telemetry"].keys(), [v.numpy() for v in sample["telemetry"].values()]))
            # update cam2world functions with new telemetry:
            self.cam2world.digest_new_telemetry(sample["telemetry"])
            ori_im = sample["image"].numpy().squeeze(0)


            ''' ************************************** '''
            tracks, detections = self.deepsort_core(ori_im)
            ''' ************************************** '''

            tracking_time = time.time()

            # draw boxes for visualization
            if len(detections) > 0 and args.debug_mode and args.display:
                ori_im = draw_boxes(ori_im,
                                    [detection.to_tlbr() for detection in detections],
                                    clr=[255, 255, 0])

                for detection in detections:
                    ax_map.scatter(detection.utm_pos[0], detection.utm_pos[1],
                                   marker='x', color='yellow', alpha=0.5)

            if self.args.display and args.debug_mode:
                ax_map.scatter(sample["telemetry"]["utmpos"][0], sample["telemetry"]["utmpos"][1],
                               marker='o', color='b', alpha=0.5)

                ax_radar.cla()
                ax_radar.set_theta_direction(-1)
                ax_radar.set_rlabel_position(90)
                ax_radar.set_rlim(bottom=0, top=20)
                ax_radar.set_theta_zero_location(
                    "N")  # , offset=+np.rad2deg(sample["telemetry"]["yaw_pitch_roll"][0][0]))

                confidence = []
                track_id = []
                cls_names = []
                utm_pos = []
                bbox_tlbr = []

                for ind, track in enumerate(tracks):
                    confidence.append(track.confidence)
                    track_id.append(track.track_id)
                    cls_names.append(self.cls_dict[track.cls_id])
                    utm_pos.append(track.utm_pos)
                    r0, c0 = self.cam2world.convert_utm_coordinates_to_bbox_center(track.utm_pos)
                    tlbr = np.array([c0[0] - track.bbox_width/2, r0[0] - track.bbox_height/2,
                                      c0[0] + track.bbox_width/2 + 1, r0[0] + track.bbox_height/2 + 1])
                    bbox_tlbr.append(tlbr)
                    ax_map.scatter(track.utm_pos[0], track.utm_pos[1], marker='^',
                                   color='r', alpha=0.5)

                    relxyz = self.cam2world.convert_bbox_tlbr_to_relative_to_camera_xyz(tlbr)
                    Rz = relxyz[1]
                    Rx = relxyz[0]
                    R = np.sqrt(Rz ** 2 + Rx ** 2)
                    theta_deg = np.arctan2(Rx, Rz)
                    ax_radar.scatter(theta_deg,R)# + sample["telemetry"]["yaw_pitch_roll"][0][0], R)

                ori_im = draw_boxes(ori_im,
                                    bbox_tlbr,
                                    confs=confidence,
                                    target_id=track_id,
                                    target_xyz=utm_pos,
                                    cls_names=cls_names,
                                    clr=[255, 0, 0])


            cv2.imshow("cam0", cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB))
            cv2.waitKey(1)


            plt.pause(0.05)
            # time.sleep(0.001)
            end_time = time.time()
            print(
                'frame: {}, total time: {:.3f}[msec], tracking time: {:.3f}[msec], visualize time: {:.3f}[msec]'.format(
                    frame, (end_time - start_time) * 1E3, (tracking_time - start_time) * 1E3,
                    (end_time - tracking_time) * 1E3))
            frame += 1

        if self.args.save_path:
            self.writer.release()

        plt.ioff()

    def deepsort_core(self, img):
        ''' *** THIS IS WHERE THE MAGIC HAPPENS *** '''
        # do detection:
        detections = self.DeepSort.detect(img)

        # udpate detections with utm coordinates using cam2world
        for detection in detections:
            detection.update_positions_using_telemetry(self.cam2world)

        # associate tracks with detections
        tracks = self.DeepSort.track(detections, timestamp=self.cam2world.telemetry["timestamp"][0])

        return tracks, detections

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, default='')
    parser.add_argument("--data", type=str, help='folder containing all recorded data')
    parser.add_argument("--batch-size", type=int, default=1, help='folder containing all recorded data')
    parser.add_argument("--config_deepsort", type=str, default="./configs/deep_sort.yaml")
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--config_sensors", type=str, default="./configs/sensors.yaml")
    parser.add_argument("--display_width", type=int, default=800)
    parser.add_argument("--display_height", type=int, default=600)
    parser.add_argument("--save_path", type=str, default="")#"./demo/demo.avi")
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
    # cfg.merge_from_file(args.config_detection)
    cfg.merge_from_file(args.config_deepsort)

    torch.set_num_threads(cfg.NUM_CPU_CORES)
    print("Using {} CPU cores".format(torch.get_num_threads()))

    tracker = Tracker(cfg, args)
    tracker.run()

    plt.show()
