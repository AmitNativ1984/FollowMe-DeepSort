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