import numpy as np
from math import pi

class Cam2World(object):
    def __init__(self, imgWidth, imgHeight, angFovX, obj_height_meters=1.8, cam_yaw_pitch_roll=[[0], [0], [0]], cam_pos=[0, 0]):
        """ angFovY, angFovX: angular field of view of camera in vertical and horizontal directions
            COORDINATES CONVERSION: (X, Y, Z) = [HORIZONTAL, DEPTH, HEIGHT]
            COORDINATES CONVERSION UTM: (LAT, LONG, HEIGHT) = (E, N, HEIGHT)
        """


        self.H = imgHeight
        self.W = imgWidth
        self.THETA_X = pi / 180.0 * angFovX   # converting to radians
        self.camAngle = np.array(cam_yaw_pitch_roll)
        self.cam_position_rel_to_imu = np.array(cam_pos)
        # focal point of camera, in pixels
        self.F0 = (self.W / 2.0) / np.tan(self.THETA_X / 2)
        self.obj_height_meters = obj_height_meters

    def digest_new_telemetry(self, telemetry):
        """" update all matrices with new telemerty data """
        yaw = telemetry["yaw_pitch_roll"][0][0]

        def remove_360_deg_periods(deg):
            while deg > 360:
                deg -= 360
            while deg < -360:
                deg += 360
            return deg

        telemetry["yaw_pitch_roll"][0] = [remove_360_deg_periods(deg) for deg in telemetry["yaw_pitch_roll"][0]]

        # convert deg to rad:
        telemetry["yaw_pitch_roll"][0] *= np.pi / 180

        self.telemetry = telemetry
        self.telemetry["utmpos"] = self.telemetry["utmpos"].transpose()

        self.R_vehicle, self.R_cam = self.calc_rotation_matrices()
        # self.T = self.calc_cam_translation()

    def calc_rotation_matrices(self):
        yaw, pitch, roll = self.telemetry["yaw_pitch_roll"][0]

        def get_3D_rotation_mat(yaw, pitch, roll):

            # rotate cam according to robot yaw
            R_yaw = np.array([[np.cos(yaw),  np.sin(yaw), 0],
                              [-np.sin(yaw), np.cos(yaw), 0],
                              [0,            0,           1]])

            R_roll  = np.array([[np.cos(pitch),  0.,    np.sin(pitch)],
                                [0.,             1.,               0.],
                                [-np.sin(pitch), 0.,    np.cos(pitch)]])

            R_pitch = np.array([[1.,              0.,                          0.],
                                [0.,              np.cos(roll),      np.sin(roll)],
                                [0.,             -np.sin(roll),      np.cos(roll)]])

            return R_yaw @ R_pitch @ R_roll



        # rotate vehicle coordinates to world coordinates
        R_vehicle = get_3D_rotation_mat(yaw, pitch, roll)

        # rotate cam coordinates to vehicle coordinates
        R_cam = get_3D_rotation_mat(self.camAngle[0, 0], self.camAngle[1, 0], self.camAngle[2, 0])

        return R_vehicle, R_cam

    def pix2angle(self, col, row):
        """ returns the vetical and horizontal angles of a ray transmitted through pixel (y,x)"""

        h = self.H / 2 - row
        w = -(self.W / 2 - col)

        theta_x = np.arctan2(w, self.F0)
        theta_y = np.arctan2(h, self.F0)

        return theta_x, theta_y

    def convert_bbox_tlbr_to_xyz_rel2cam(self, bbox_tlbr):
        """ returns relative coordinates from camera center (horizontal,depth,height) = (x,y,z)
            (x1,y1): top left coordinates in pixels
            (x2,y2): bottom right coordinates in pixels
            obj_height_meters: object height in meters
        """

        x1, y1, x2, y2 = bbox_tlbr

        theta_x_TopLeft, theta_y_TopLeft = self.pix2angle(x1, y1)
        theta_x_BottomRight, theta_y_BottomRight = self.pix2angle(x2, y2)

        h0 = self.obj_height_meters

        # calculating distance to object in meters: Y/H0 = y/h0 = 1/tan(theta_y1 - theta_y2)
        y = h0 / (np.tan(theta_y_TopLeft) - np.tan(theta_y_BottomRight))

        # calculating angle to bbox center:
        x0_ps = np.mean([x1, x2])
        y0_ps = np.mean([y1, y2])

        theta_x0, theta_z0 = self.pix2angle(x0_ps, y0_ps)

        x = y * np.tan(theta_x0)
        z = y * np.tan(theta_z0)

        return np.array([x, y, z])  # (horizontal, depth, height)

    def convert_bbox_tlbr_to_utm(self, bbox_tlbr):

        xyz_rel2cam = np.array([self.convert_bbox_tlbr_to_xyz_rel2cam(bbox_tlbr)]).transpose() #(x, height, depth)

        # rotate camera relative to robot 12 O'clock and translate coordinates to IMU
        xyz_rel2imu = self.R_cam @ xyz_rel2cam + self.cam_position_rel_to_imu

        # rotate target xyz position relative to north (yaw angle) and add telemetry to get total utm pos
        """ utm_pos is [long, lat, height] = [x,y,z] """
        utm_pos = self.R_vehicle @ xyz_rel2imu + self.telemetry["utmpos"]

        return utm_pos

    def convert_xyz_rel2cam_to_bbox_center(self, xyz_rel2cam):
        horizontal_pos, depth, vertical_pos = xyz_rel2cam

        col_center = horizontal_pos / depth * self.F0 + self.W / 2
        row_center = -vertical_pos / depth * self.F0 + self.H / 2


        # col_center = x / y * self.F0 + self.W / 2
        # row_center = -z / y * self.F0 + self.H / 2

        height = self.obj_height_meters / depth * self.F0

        return row_center, col_center, height

    def convert_utm_coordinates_to_xyz_rel2imu(self, utm_pos):
        # from utm coordinates to xyz coordinates relative to imu:
        return np.linalg.inv(self.R_vehicle) @ (utm_pos - self.telemetry["utmpos"])

    def convert_xyz_rel2imu_to_xyz_rel2cam(self, xyz_rel2imu):
        # from xyz rel2imu and aligned to vehicle 12 O'clock, to xyz_rel2cam
        return np.linalg.inv(self.R_cam) @ (xyz_rel2imu - self.cam_position_rel_to_imu)

    def convert_utm_coordinates_to_xyz_rel2cam(self, utm_pos):
        # from utm coordinates to xyz coordinates relative to imu:
        xyz_rel2imu = self.convert_utm_coordinates_to_xyz_rel2imu(utm_pos)

        # from xyz rel2imu and aligned to vehicle 12 O'clock, to xyz_rel2cam
        xyz_rel2cam = self.convert_xyz_rel2imu_to_xyz_rel2cam(xyz_rel2imu)

        return xyz_rel2cam

    def is_utm_coordinates_in_cam_FOV(self, utm_pos):
        xyz_rel2cam = self.convert_utm_coordinates_to_xyz_rel2cam(utm_pos)
        angle = np.arctan2(xyz_rel2cam[0], xyz_rel2cam[1]) * 180 / np.pi

        if xyz_rel2cam[1] >= 0: #and np.abs(angle) <= self.THETA_X/2 * 180 / np.pi:
            return True
        else:
            return False

    def convert_utm_coordinates_to_bbox_center(self, utm_pos):
        xyz_rel2cam = self.convert_utm_coordinates_to_xyz_rel2cam(utm_pos)

        row_center, col_center, height = self.convert_xyz_rel2cam_to_bbox_center(xyz_rel2cam)

        return row_center, col_center, height

    def get_target_relative_xyz(self, robot_utm, target_utm):
        relative_xyz = -robot_utm + target_utm

        return relative_xyz

    def project_xyz_in_local_camera_coordinates_to_pixels(self, xyz):
        horizontal_pos, depth, vertical_pos = xyz

        cols = horizontal_pos / depth * self.F0 + self.W / 2
        rows = vertical_pos / depth * self.F0 + self.H / 2

        return rows, cols

    def convert_target_utm_relative_xyz(self, robot_utm, target_utm, robot_yaw, camera_angle=0,
                                        camera_pos_on_vehicle=np.array([[0], [1.7]])):
        target_relative_pos = self.get_target_relative_xyz(robot_utm, target_utm)
        RotationMat = self.compute_yaw2camera_rotation_matrix(robot_yaw, camera_angle)

        target_xyz = np.matmul(RotationMat, target_relative_pos) - camera_pos_on_vehicle
        return target_xyz

    def convert_xyz_rel2cam_to_bbox(self, target_xyz, org_bbox_tlwh):
        row, col = self.project_xyz_in_local_camera_coordinates_to_pixels(target_xyz)

        return row, col
