import argparse
from sys import platform

from .models import *  # set ONNX_EXPORT in models.py
from .utils.datasets import *
from .utils.utils import *
from ..YOLOv3.yolo_utils import xyxy_to_xywh


class YOLOv3(object):
    def __init__(self, opt):
        # todo: initi ultralytics yolov3
        self.opt = opt
        self.img_size = self.opt.img_size

        # net definition
        self.net = Darknet(opt.cfg, self.img_size)

        self.device = torch_utils.select_device(device=self.opt.device)

        # loading weights:
        if self.opt.weights.endswith('.pt'):  # pytorch format
            self.net.load_state_dict(torch.load(self.opt.weights, map_location=self.device)['model'])
        else:  # darknet format
            load_darknet_weights(self.net, self.opt.weights)

        print('Loading weights from %s... Done!' % (self.opt.weights))

        self.net.eval()
        self.net.to(self.device)

        # constants
        self.size = self.img_size, self.img_size
        self.score_thresh = self.opt.conf_thres
        self.conf_thresh = 0.01
        self.nms_thresh = self.opt.iou_thres
        self.use_cuda = 'cuda' in self.device.type
        self.is_xywh = True
        self.class_names = load_classes(opt.names)
        self.num_classes = np.shape(self.class_names)


    def __call__(self, ori_img):
        # img to tensor
        assert isinstance(ori_img, np.ndarray), "input must be a numpy array!"
        img = ori_img.astype(np.float) / 255.

        img = cv2.resize(img, self.size)
        img = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0)

        # forward
        with torch.no_grad():
            img = img.to(self.device)
            pred = self.net(img)[0]

            boxes = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, classes=self.opt.classes,
                                       agnostic=self.opt.agnostic_nms)

        height, width = ori_img.shape[:2]
        # for i, det in enumerate(boxes):  # detections per image
        #     if det is not None and len(det):
        #         # Rescale boxes from img_size to im0 size
        #         det[:, 0] = det[:, 0] / img.shape[1]
        #         det[:, 1] = det[:, 1] / img.shape[0]
        #         det[:, 2] = det[:, 2] / img.shape[1]
        #         det[:, 3] = det[:, 3] / img.shape[0]

        # boxes = det
        if not torch.is_tensor(boxes[0]):
            if not boxes[0]:
                return [None], [None], [None]


        height, width = ori_img.shape[:2]
        try:
            bbox = boxes[0][:, :4]
        except Exception:
            print('x')

        if self.is_xywh:
            # bbox x y w h
            bbox = xyxy_to_xywh(bbox)

        bbox = bbox * torch.FloatTensor([[width/416, height/416, width/416, height/416]]).to(boxes[0].device)


        cls_conf = boxes[0][:, 4]
        cls_ids = boxes[0][:, 5]
        return [bbox.cpu().numpy()], [cls_conf.cpu().numpy()], [cls_ids.cpu().numpy()]