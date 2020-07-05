import numpy as np
from math import pi

class Cam2World(object):
    def __init__(self, imgWidth, imgHeight, angFovX):
        """ angFovY, angFovX: angular field of view of camera in vertical and horizontal directions """

        self.H = imgHeight
        self.W = imgWidth
        self.THETA_X = pi / 180.0 * angFovX   # converting to radians

        # focal point of camera, in pixels
        self.F0 = (self.W / 2.0) / np.tan(self.THETA_X / 2)

    def pix2angle(self, x, y):
        """ returns the vetical and horizontal angles of a ray transmitted through pixel (y,x)"""

        h = self.H / 2 - y
        w = -(self.W / 2 - x)

        theta_x = np.arctan2(w, self.F0)
        theta_y = np.arctan2(h, self.F0)

        return theta_x, theta_y

    def obj_world_xyz(self, x1, y1, x2, y2, obj_height_meters):
        """ returns relative coordinates from camera center (x,y,z)
            (x1,y1): top left coordinates in pixels
            (x2,y2): bottom right coordinates in pixels
            obj_height_meters: object height in meters
        """

        theta_x_TopLeft, theta_y_TopLeft = self.pix2angle(x1, y1)
        theta_x_BottomRight, theta_y_BottomRight = self.pix2angle(x2, y2)

        h0 = obj_height_meters

        # calculating distance to object in meters: Z/H0 = z/h0 = 1/tan(theta_y1 - theta_y2)
        z = h0 / (np.tan(theta_y_TopLeft) - np.tan(theta_y_BottomRight))

        # calculating angle to bbox center:
        x0_ps = np.mean([x1, x2])
        y0_ps = np.mean([y1, y2])

        theta_x0, theta_y0 = self.pix2angle(x0_ps, y0_ps)

        x = z * np.tan(theta_x0)
        y = z * np.tan(theta_y0)

        return [x, y, z]

    def get_target_relative_xyz(self, robot_utm, target_utm):
        relative_xyz = -robot_utm + target_utm

        return relative_xyz

    def compute_yaw2camera_rotation_matrix(self, robot_yaw, camera_angle=0):
        """ robot_yaw and camera_angle are give in angles!! [o]"""
        while robot_yaw > 360:
            robot_yaw -= 360
        while camera_angle > 360:
            camera_angle -= 360

        robot_yaw = np.deg2rad(robot_yaw)
        camera_angle = np.deg2rad(camera_angle)

        # rotation from yaw angle back to 0 degress
        R1 = np.array([[np.cos(robot_yaw), -np.sin(robot_yaw)],
                      [np.sin(robot_yaw), np.cos(robot_yaw)]])

        # rotating from 0 degrees to camera angle of sight
        R2 = np.array([[np.cos(camera_angle), np.sin(camera_angle)],
                       [-np.sin(camera_angle), np.cos(camera_angle)]])

        RotationMat = np.matmul(R2, R1)

        return RotationMat

    def project_xyz_in_local_camera_coordinates_to_pixels(self, xyz):
        # todo: add rows too!!!
        x = xyz[0]
        # y = xyz[1]
        # z = xyz[2]
        z = xyz[1]

        cols = x / z * self.F0 + self.W / 2
        # rows = y / z * self.F0 + self.H / 2

        return cols #rows, cols

    def convert_target_utm_relative_xyz(self, robot_utm, target_utm, robot_yaw, camera_angle=0,
                                        camera_pos_on_vehicle=np.array([[0], [1.7]])):
        target_relative_pos = self.get_target_relative_xyz(robot_utm, target_utm)
        RotationMat = self.compute_yaw2camera_rotation_matrix(robot_yaw, camera_angle)

        target_xyz = np.matmul(RotationMat, target_relative_pos) - camera_pos_on_vehicle
        return target_xyz

    def xyz2rowcol(self, target_xyz):
        # todo: add rows too!!!!
        col = self.project_xyz_in_local_camera_coordinates_to_pixels(target_xyz)

        return col, #row, col