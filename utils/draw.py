import numpy as np
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib as mpl


palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)


def compute_color_for_labels(label, target_id):
    """
    Simple function that adds fixed color depending on the class
    """
    if target_id == None:
        color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    else:
        if label == target_id:
            color = [0, 255, 0]
        else:
            color = [0, 0, 255]
    return tuple(color)


def draw_boxes(img, bbox, identities=None, offset=(0,0), confs=None, target_id=None, target_xyz=None, cls_names=None,
               clr=None):
    for i,box in enumerate(bbox):
        x1,y1,x2,y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        if clr==None:
            color = compute_color_for_labels(id, target_id)
        else:
            color=clr

        if (identities) is not None:
            label = '{}{:d}'.format("", id)
        else:
            label = ""

        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2 , 2)[0]
        cv2.rectangle(img,(x1, y1),(x2,y2),color,3)
        cv2.rectangle(img,(x1, y1),(x1+t_size[0]+3,y1+t_size[1]+4), color,-1)
        cv2.putText(img,label,(x1,y1+t_size[1]+4), cv2.FONT_HERSHEY_PLAIN, 2, [255,255,255], 2)
        if confs:
            xyz = target_xyz[i].squeeze(1)
            t_size = cv2.getTextSize('{}({:.2f}, {:.2f}, {:.2f})[m]'.format(" ", xyz[0], xyz[1], xyz[2]), cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
            cv2.rectangle(img, (x1, y2), (x1 + t_size[0] + 3,  y2 + t_size[1] * 7), color, 1)
            cv2.putText(img, '{}{:.3f}'.format("conf:", confs[i]), (x1, y2 + 2 * t_size[1]), cv2.FONT_HERSHEY_PLAIN, 2, [255,255,255], 2)
            cv2.putText(img, '{}'.format(cls_names[i]), (x1, y2 + 4 * t_size[1]), cv2.FONT_HERSHEY_PLAIN, 2, [255,255,255], 2)
            cv2.putText(img, '({:.2f}, {:.2f}, {:.2f})[m]'.format(xyz[0], xyz[1], xyz[2]),
                        (x1, y2 + 6 * t_size[1]), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)

    return img

def create_radar_plot(xlim=(-25, 25), ylim=(-25, 25), radar_fig=None, ax_polar=None, ax_carthesian=None):
    # setting up canvas for polar and cartesian plot:

    if radar_fig is None:
        radar_fig = plt.figure()

    if ax_polar is not None:
        ax_polar.clear()

    if ax_carthesian is not None:
        ax_carthesian.clear()

    # setting the axis limits in [left, bottom, width, height]
    rect = [0.1, 0.1, 0.8, 0.8]

    # the carthesian axis:
    ax_carthesian = radar_fig.add_axes(rect)
    ax_carthesian.set_xlim(xlim)
    ax_carthesian.set_ylim(ylim)
    ax_carthesian.set_aspect('equal')

    # the polar axis:
    ax_polar = radar_fig.add_axes(rect, polar=True, frameon=False)
    ax_polar.set_theta_direction(-1)
    ax_polar.set_theta_zero_location("N")
    ax_polar.set_rlabel_position(90)
    ax_polar.set_ylim(0, max(ylim))


    ticklabelpad = mpl.rcParams['xtick.major.pad']


    ax_carthesian.annotate('[m]', xy=(1, 0), xytext=(5, -ticklabelpad), ha='left', va='top',
                xycoords='axes fraction', textcoords='offset points')

    ticklabelpad = mpl.rcParams['ytick.major.pad']
    ax_carthesian.annotate('[m]', xy=(0, 1), xytext=(-5, ticklabelpad), ha='right', va='top',
                           xycoords='axes fraction', textcoords='offset points')

    ax_polar.set_rmax(max(ylim))
    plt.rgrids((5, 10, 15, 20))
    ax_polar.grid(True)

    return radar_fig, ax_polar, ax_carthesian

if __name__ == '__main__':
    for i in range(82):
        print(compute_color_for_labels(i))
