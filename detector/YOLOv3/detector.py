import torch
# import time
import numpy as np
import cv2
import torchvision
from .darknet import Darknet
from .yolo_utils import get_all_boxes, nms, post_process, xywh_to_xyxy, xyxy_to_xywh
from .nms import boxes_nms


class YOLOv3(object):
    def __init__(self, cfgfile, weightfile, namesfile, score_thresh=0.7, conf_thresh=0.01, nms_thresh=0.45, is_xywh=False, use_cuda=True):
        # net definition
        self.net = Darknet(cfgfile)
        self.net.load_weights(weightfile)
        print('Loading weights from %s... Done!' % (weightfile))
        self.device = "cuda" if use_cuda else "cpu"
        self.net.eval()
        self.net.to(self.device)

        # constants
        self.size = self.net.width, self.net.height
        self.score_thresh = score_thresh
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.use_cuda = use_cuda
        self.is_xywh = is_xywh
        self.num_classes = self.net.num_classes
        self.class_names = self.load_class_names(namesfile)

    def __call__(self, ori_img):
        # img to tensor
        assert isinstance(ori_img, np.ndarray), "input must be a numpy array!"
        img = ori_img.astype(np.float)/255.

        img = cv2.resize(img, self.size)
        img = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0)
        batch_size = img.shape[1] // 3
        img = img.view(batch_size, 3, img.shape[2], img.shape[3])
        # forward
        with torch.no_grad():
            img = img.to(self.device)
            out_boxes = self.net(img)
            boxes = get_all_boxes(out_boxes, self.conf_thresh, self.num_classes, use_cuda=self.use_cuda) #batch size is 1
            # boxes = nms(boxes, self.nms_thresh)

            boxes = post_process(boxes, self.net.num_classes, self.conf_thresh, self.nms_thresh)

        height, width = ori_img.shape[:2]
        denormalize_bbox_tensor = torch.FloatTensor([[width, height, width, height]]).type_as(boxes[0])

        bbox = []
        cls_conf = []
        cls_ids = []
        for b in range(batch_size):
            boxes[b] = boxes[b][boxes[b][:, -2] > self.score_thresh] # bbox xmin ymin xmax ymax

        # if len(boxes)==0:
        #     return None,None,None


            bbox.append(boxes[b][:, :4])
            if bbox[b].nelement() != 0:
                if self.is_xywh:
                # bbox x y w h
                    bbox[b] = xyxy_to_xywh(bbox[b])


                bbox[b] = (bbox[b] * denormalize_bbox_tensor).cpu().numpy()

            cls_conf.append((boxes[b][:, 5]).cpu().numpy())
            cls_ids.append((boxes[b][:, 6].long()).cpu().numpy())

        return bbox, cls_conf, cls_ids

    def load_class_names(self,namesfile):
        with open(namesfile, 'r', encoding='utf8') as fp:
            class_names = [line.strip() for line in fp.readlines()]
        return class_names


class YOLOv3_TINY(object):
    def __init__(self, cfgfile, weightfile, namesfile, score_thresh=0.7, conf_thresh=0.01, nms_thresh=0.45,
                 is_xywh=False, use_cuda=True):
        # todo: change to yolov3-tiny
        # net definition
        self.net = Darknet(cfgfile)
        self.net.load_weights(weightfile)
        print('Loading weights from %s... Done!' % (weightfile))
        self.device = "cuda" if use_cuda else "cpu"
        self.net.eval()
        self.net.to(self.device)

        # constants
        self.size = self.net.width, self.net.height
        self.score_thresh = score_thresh
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.use_cuda = use_cuda
        self.is_xywh = is_xywh
        self.num_classes = self.net.num_classes
        self.class_names = self.load_class_names(namesfile)

    def __call__(self, ori_img):
        # img to tensor
        assert isinstance(ori_img, np.ndarray), "input must be a numpy array!"
        img = ori_img.astype(np.float) / 255.

        img = cv2.resize(img, self.size)
        img = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0)

        # forward
        with torch.no_grad():
            img = img.to(self.device)
            out_boxes = self.net(img)
            boxes = get_all_boxes(out_boxes, self.conf_thresh, self.num_classes,
                                  use_cuda=self.use_cuda)  # batch size is 1
            # boxes = nms(boxes, self.nms_thresh)

            boxes = post_process(boxes, self.net.num_classes, self.conf_thresh, self.nms_thresh)[0].cpu()
            boxes = boxes[boxes[:, -2] > self.score_thresh, :]  # bbox xmin ymin xmax ymax

        if len(boxes) == 0:
            return None, None, None

        height, width = ori_img.shape[:2]
        bbox = boxes[:, :4]
        if self.is_xywh:
            # bbox x y w h
            bbox = xyxy_to_xywh(bbox)

        bbox = bbox * torch.FloatTensor([[width, height, width, height]])
        cls_conf = boxes[:, 5]
        cls_ids = boxes[:, 6].long()
        return bbox.numpy(), cls_conf.numpy(), cls_ids.numpy()

    def load_class_names(self, namesfile):
        with open(namesfile, 'r', encoding='utf8') as fp:
            class_names = [line.strip() for line in fp.readlines()]
        return class_names


def demo():
    import os
    from vizer.draw import draw_boxes

    yolo = YOLOv3("cfg/yolov3_probot.cfg", "weight/yolov3_ultralytics.pt", "cfg/kitti_probot.names")
    print("yolo.size =",yolo.size)
    root = "/home/amit/Data/Vehicle_Recordings/Ben_Shemen/14-01-2020/day/drive3-day/images"
    resdir = os.path.join(root, "results")
    os.makedirs(resdir, exist_ok=True)
    files = [os.path.join(root,file) for file in os.listdir(root) if file.endswith('.jpg') | file.endswith('.bmp') ]
    files.sort()
    for filename in files:
        img = cv2.imread(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        bbox, cls_conf, cls_ids = yolo(img)

        if bbox is not None:
            img = draw_boxes(img, bbox, cls_ids, cls_conf, class_name_map=yolo.class_names)
        # save results
        cv2.imwrite(os.path.join(resdir, os.path.basename(filename)) ,img[:,:,(2,1,0)])
        # imshow
        # cv2.namedWindow("yolo", cv2.WINDOW_NORMAL)
        # cv2.resizeWindow("yolo", 600,600)
        # cv2.imshow("yolo",res[:,:,(2,1,0)])
        # cv2.waitKey(0)


if __name__ == "__main__":
    demo()
