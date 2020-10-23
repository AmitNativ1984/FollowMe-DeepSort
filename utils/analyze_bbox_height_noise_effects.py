from utils.camera2world import Cam2World
import matplotlib.pyplot as plt
import numpy as np

def run_over_bbox_size(cam2world, factor, obj_height_meters):
    bbox_height_px = 700
    h = []
    z = []
    z0 = []
    while bbox_height_px > 50:

        bbox_noise = np.array(range(-20, 20, 5))
        t = 10
        l = 10
        b = t + bbox_height_px
        r = l + 50
        x0y0z0 = cam2world.obj_world_xyz(l, t, r, b, obj_height_m)
        zmin = np.inf
        zmax = 0
        for noise in bbox_noise:
            h.append(bbox_height_px)
            b = t + bbox_height_px + noise

            xyz = cam2world.obj_world_xyz(l, t, r, b, obj_height_m)
            z.append(xyz[-1])
            if zmin > z[-1]:
                zmin = z[-1]
            if zmax < z[-1]:
                zmax = z[-1]

            z0.append(x0y0z0[-1])

        bbox_height_px -= factor

    return np.array(h), np.array(z0), np.array(z)

if __name__ == '__main__':
    obj_height_m = 1.8
    img_width = 1280
    img_height = 720
    camera_fov_x = 77.04
    delta = 10
    cam2world = Cam2World(img_width, img_height, camera_fov_x)

    plt.figure
    plt.xlabel('z [m]')
    plt.ylabel('z [m]')

    h1, z0, z1 = run_over_bbox_size(cam2world, delta, obj_height_m)
    plt.scatter(z0, z1, label='bbox height change: [-20, 20, 5]')
    plt.plot(z0, z0, color='red')

    # h2, z2 = run_over_bbox_size(cam2world, delta, obj_height_m)
    # plt.plot(h2, z2, color='red', label='20% change')

    plt.legend()

    dh1 = h1[1:] - h1[:-1]
    dz1 = z1[1:] - z1[:-1]
    # plotting derivatives:

    plt.figure()
    plt.plot(dh1, dz1)

    plt.show()