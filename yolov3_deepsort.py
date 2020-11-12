import os
import cv2
import time
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from deep_sort.dataloaders.vehicle_sensors_dataloader import ProbotSensorsDataset
from utils.draw import draw_boxes, create_radar_plot
from utils.parser import get_config
from yolov3_deepsort_headless import Tracker

class DeepSortManager(object):
    def __init__(self, tracker, args):
        self.args = args
        self.tracker = tracker
        self.dataset = ProbotSensorsDataset(self.args)
        self.data_loader = torch.utils.data.DataLoader(self.dataset, batch_size=1, shuffle=False)
        self.frame = 0

        self.vdo = cv2.VideoCapture()

    def __enter__(self):
        assert os.path.isdir(self.args.data), "Error: path error"

        self.im_width = 1280
        self.im_height = 720

        if self.args.save_path:
            fourcc =  cv2.VideoWriter_fourcc(*'XVID')
            self.writer = cv2.VideoWriter(self.args.save_path, fourcc, 20, (self.im_width,self.im_height))

        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)

        if self.args.save_path:
            self.writer.release()

    def create_figures(self):
        # set figures and plots:
        fig3Dtracking = plt.figure(figsize=(10, 4))
        plt.get_current_fig_manager().window.wm_geometry("+0+0")

        # to put it into the upper left corner for example:
        # utm map
        ax_map = fig3Dtracking.add_subplot(1, 2, 1)
        ax_map.set_title("UTM")
        ax_map.set_aspect('equal', 'box')
        ax_map.grid(True)

        # vehicle radar
        ax_radar = fig3Dtracking.add_subplot(1, 2, 2, projection='polar')
        ax_radar.set_aspect('equal', 'box')
        ax_radar.set_theta_direction(-1)
        ax_radar.set_theta_zero_location("N")
        ax_radar.set_rlabel_position(90)
        ax_radar.set_rlim(bottom=0, top=30)

        fig3Dtracking.suptitle('UTM Tracking')
        cv2.namedWindow('cam0', cv2.WINDOW_NORMAL)

        return ax_map, ax_radar

    def run(self):
        U = np.array([[0], [0], [0], [0]])
        ax_map, ax_radar = self.create_figures()
        plt.ion()

        for i, sample in enumerate(self.data_loader):
            start_time = time.time()
            telemetry = dict(zip(sample["telemetry"].keys(), [v.numpy() for v in sample["telemetry"].values()]))

            ori_im = sample["image"].numpy().squeeze(0)

            ''' ************************************** '''
            tracks, detections = self.tracker.run_tracking(ori_im, telemetry)
            ''' ************************************** '''

            tracking_time = time.time()

            if self.args.display and args.debug_mode:
                self.display(tracks, detections, telemetry, ori_im, ax_radar, ax_map)
            end_time = time.time()
            print(
                'frame: {}, total time: {:.3f}[msec], tracking time: {:.3f}[msec], visualize time: {:.3f}[msec]'.format(
                    self.frame, (end_time - start_time) * 1E3, (tracking_time - start_time) * 1E3,
                                (end_time - tracking_time) * 1E3))
            self.frame += 1

    def display(self, tracks, detections, telemetry, ori_im, ax_radar, ax_map):
        # draw boxes for visualization
        if len(detections) > 0:
            ax_radar.cla()
            ax_radar.set_theta_direction(-1)
            ax_radar.set_rlim(bottom=0, top=30)
            ax_radar.set_theta_zero_location(
                "N")

            ori_im = draw_boxes(ori_im,
                                [detection.to_tlbr() for detection in detections],
                                target_id=[detection.cls_id for detection in detections],
                                clr=[50, 200, 0])

            ax_map.cla()
            ax_map.grid(True)
            ax_map.set_title("UTM")
            for detection in detections:
                ax_map.scatter(detection.utm_pos[0], detection.utm_pos[1],
                               marker='+', color='g')

                relxyz = self.tracker.cam2world.convert_bbox_tlbr_to_xyz_rel2cam(detection.to_tlbr())
                Rz = relxyz[1]
                Rx = relxyz[0]
                R = np.sqrt(Rz ** 2 + Rx ** 2)
                theta_deg = np.arctan2(Rx, Rz)
                ax_radar.scatter(theta_deg, R, color='g', marker='x')

        if self.args.display and args.debug_mode:
            ax_map.scatter(telemetry["utmpos"][0], telemetry["utmpos"][1],
                           marker='o', color='b', alpha=0.5)
            ax_map.axis(xmin=telemetry["utmpos"][0]-30, xmax=telemetry["utmpos"][0] + 30,
                        ymin=telemetry["utmpos"][1] - 30, ymax=telemetry["utmpos"][1] + 30)


            confidence = []
            track_id = []
            cls_names = []
            utm_pos = []
            bbox_tlbr = []

            for ind, track in enumerate(tracks):
                utm_pos.append(track.mean)
                r0, c0, h = self.tracker.cam2world.convert_utm_coordinates_to_bbox_center(track.mean[:3])
                # tlbr = np.array([c0[0] - track.bbox_width/2, r0[0] - track.bbox_height/2,
                #                   c0[0] + track.bbox_width/2 + 1, r0[0] + track.bbox_height/2 + 1])
                tlbr = track.utm_to_bbox_tlbr(self.tracker.cam2world)
                if track.in_cam_FOV:
                    bbox_tlbr.append(tlbr)
                    confidence.append(track.confidence)
                    track_id.append(track.track_id)
                    cls_names.append(self.tracker.cls_dict[track.cls_id])

                ax_map.scatter(track.mean[0], track.mean[1], marker='o',
                               color='r', s=5)
                ax_map.annotate(str(track.track_id), xy=(track.mean[0], track.mean[1]), xycoords='data')

                relxyz = self.tracker.cam2world.convert_utm_coordinates_to_xyz_rel2cam(track.mean[:3])
                Rz = relxyz[1]
                Rx = relxyz[0]
                R = np.sqrt(Rz ** 2 + Rx ** 2)
                theta_deg = np.arctan2(Rx, Rz)
                ax_radar.scatter(theta_deg, R, color='r', alpha=0.5)# + sample["telemetry"]["yaw_pitch_roll"][0][0], R)
                ax_radar.annotate(str(track.track_id), xy=(theta_deg, R), xycoords='data')

                lambda1, lambda2 = track.cov_eigenvalues
                mue1, mue2 = track.cov_eigenvectors
                chi2inv95dim2 = 5.9915
                r1 = np.sqrt(chi2inv95dim2 * lambda1)
                r2 = np.sqrt(chi2inv95dim2 * lambda2)
                angle = np.arctan2(mue1[1], mue1[0]) * 180 / np.pi

                ellipse = Ellipse((Rx, Rz), r1, r2, angle=angle, transform=ax_radar.transData._b, color="blue", facecolor=None, fill=False)
                ax_radar.add_artist(ellipse)

                ellipse = Ellipse((track.mean[0], track.mean[1]), r1*2, r2*2, angle=angle, color="red", facecolor=None, fill=False)
                ax_map.add_artist(ellipse)

            ori_im = draw_boxes(ori_im,
                                bbox_tlbr,
                                confs=confidence,
                                target_id=track_id,
                                target_xyz=utm_pos,
                                cls_names=cls_names,
                                clr=[255, 0, 0])

        cv2.imshow("cam0", cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB))
        cv2.resizeWindow('cam0', int(1280/2), int(720/2))
        cv2.waitKey(1)
        plt.draw()
        plt.pause(0.001)


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

    # creating tracker:
    cls_dict = {0: 'person', 2: 'vehicle', 7: 'vehicle'}
    safezone = [100, 100, 1000, 600]
    tracker = Tracker(args.img_width, args.img_height, cls_dict, safezone,
                      args.thetaX, args.target_height)

    with DeepSortManager(tracker, args) as ds_tracker:
        ds_tracker.run()
