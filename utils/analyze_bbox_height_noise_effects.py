from utils.camera2world import Cam2World
import matplotlib.pyplot as plt
import numpy as np

def run_over_bbox_size(cam2world, factor, obj_height_meters):
    bbox_height_px = 700
    h = []
    z = []
    while bbox_height_px > 50:


        bbox_noise = np.array(range(-20, 20, 5))
        t = 10
        l = 10
        b = t + bbox_height_px
        r = l + 50
        for noise in bbox_noise:
            h.append(bbox_height_px)
            b = t + bbox_height_px + noise

            xyz = cam2world.obj_world_xyz(l, t, r, b, obj_height_m)
            z.append(xyz[-1])

        bbox_height_px -= factor

    return np.array(h), np.array(z)

if __name__ == '__main__':
    obj_height_m = 1.8
    img_width = 1280
    img_height = 720
    camera_fov_x = 77.04
    delta = 20
    cam2world = Cam2World(img_width, img_height, camera_fov_x)

    plt.figure
    plt.xlabel('bbox height [px]')
    plt.ylabel('z [m]')

    h1, z1 = run_over_bbox_size(cam2world, delta, obj_height_m)
    plt.scatter(h1, z1, label='range(-20, 20, 5)')

    # h2, z2 = run_over_bbox_size(cam2world, delta, obj_height_m)
    # plt.plot(h2, z2, color='red', label='20% change')

    plt.legend()

    dh1 = h1[1:] - h1[:-1]
    dz1 = z1[1:] - z1[:-1]
    # plotting derivatives:

    plt.figure()
    plt.plot(dh1, dz1)

    plt.show()