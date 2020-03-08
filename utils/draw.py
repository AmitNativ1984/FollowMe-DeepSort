import numpy as np
import cv2
from vizer.draw import draw_boxes

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


def draw_boxes(img, bbox, identities=None, offset=(0,0), confs=None, target_id=None, target_xyz=None, cls_names=None):
    for i,box in enumerate(bbox):
        x1,y1,x2,y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        color = compute_color_for_labels(id, target_id)
        if (identities) is not None:
            label = '{}{:d}'.format("", id)
        else:
            label = ""

        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2 , 2)[0]
        cv2.rectangle(img,(x1, y1),(x2,y2),color,3)
        cv2.rectangle(img,(x1, y1),(x1+t_size[0]+3,y1+t_size[1]+4), color,-1)
        cv2.putText(img,label,(x1,y1+t_size[1]+4), cv2.FONT_HERSHEY_PLAIN, 2, [255,255,255], 2)
        if confs:
            t_size = cv2.getTextSize('{}({:.2f}, {:.2f}, {:.2f})[m]'.format(" ", target_xyz[i][0], target_xyz[i][1], target_xyz[i][2]), cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
            cv2.rectangle(img, (x1, y2), (x1 + t_size[0] + 3,  y2 + t_size[1] * 7), color, 1)
            cv2.putText(img, '{}{:.3f}'.format("conf:", confs[i]), (x1, y2 + 2 * t_size[1]), cv2.FONT_HERSHEY_PLAIN, 2, [255,255,255], 2)
            cv2.putText(img, '{}'.format(cls_names[i]), (x1, y2 + 4 * t_size[1]), cv2.FONT_HERSHEY_PLAIN, 2, [255,255,255], 2)
            cv2.putText(img, '({:.2f}, {:.2f}, {:.2f})[m]'.format(target_xyz[i][0], target_xyz[i][1], target_xyz[i][2]),
                        (x1, y2 + 6 * t_size[1]), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)

    return img



if __name__ == '__main__':
    for i in range(82):
        print(compute_color_for_labels(i))