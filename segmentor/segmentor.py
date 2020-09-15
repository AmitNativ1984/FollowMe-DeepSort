import numpy as np
import cv2
import torch
from segmentor.pytorch_deeplab.infer_bbox import InferBoundingBox

class Segmentor(object):
    def __init__(self, cfg, cls_dict):
        # defining segmentor
        self.cfg = cfg
        self.cls_dict = cls_dict
        self.segmentor = InferBoundingBox(self.cfg.SEGMENTOR)
        self.segmentor.define_model()
        self.segmentor.load_model()

    def segment_bboxes(self, targets, im):

        bbox = [None] * len(targets)
        padded_bbox_shape = [None] * len(targets)
        org_bbox_shape = [None] * len(targets)
        bbox_resize_factor = [None] * len(targets)
        bbox_cls_id = [None] * len(targets)

        # batching all targets for inference:
        for i, track in enumerate(targets):

            x0, y0, x1, y1 = track.to_tlbr().astype(np.int)
            x0, x1 = np.clip([x0, x1], a_min=0, a_max=im.shape[1])
            y0, y1 = np.clip([y0, y1], a_min=0, a_max=im.shape[0])

            org_box = im[y0:y1, x0:x1]
            resized_bbox_tensor, padded_shape, org_shape, resize_factor = \
                self.preprocess_segmentation(org_box, self.cfg.SEGMENTOR.INPUT_SIZE_WIDTH, self.cfg.SEGMENTOR.INPUT_SIZE_HEIGHT)

            bbox[i] = resized_bbox_tensor
            padded_bbox_shape[i] = padded_shape
            org_bbox_shape[i] = org_shape
            bbox_resize_factor[i] = resize_factor
            bbox_cls_id[i] = track.cls_id


        if len(bbox) != 0:
            bbox_tensor = torch.cat(bbox).view(-1, 3, self.cfg.SEGMENTOR.INPUT_SIZE_HEIGHT, self.cfg.SEGMENTOR.INPUT_SIZE_WIDTH)
            bbox_tensor = bbox_tensor.cuda()
            org_bbox_shape = np.stack(org_bbox_shape, axis=0)

            raw_pred_mask = self.segmentor.infer(bbox_tensor)
            targets = self.postprocess_segmentation(targets, raw_pred_mask, bbox_cls_id, padded_bbox_shape, org_bbox_shape)

        return targets

    def preprocess_segmentation(self, bbox_img, W0, H0):
        H, W, C = bbox_img.shape
        org_bbox_shape = (H, W)
        padding_frame = (np.max([W, H]))
        resize_factor = W0 / padding_frame
        padded_bbox_img = cv2.copyMakeBorder(bbox_img, 0, np.abs(padding_frame -H), 0, np.abs(padding_frame -W), cv2.BORDER_CONSTANT, value=[0 ,0 ,0])
        padded_bbox_shape = padded_bbox_img.shape[:2]
        resized_bbox_image = cv2.resize(padded_bbox_img, (H0, W0))
        resized_bbox_tensor = InferBoundingBox.preprocess()(resized_bbox_image)

        return resized_bbox_tensor, padded_bbox_shape, org_bbox_shape, resize_factor

    def postprocess_segmentation(self, targets, raw_pred_mask, bbox_cls_id, padded_bbox_shape, org_bbox_shape):

        for idx in range(raw_pred_mask.shape[0]):
            mask = cv2.resize(raw_pred_mask[idx, ...], padded_bbox_shape[idx], interpolation=cv2.INTER_NEAREST)
            mask = mask[:org_bbox_shape[idx][0], :org_bbox_shape[idx][1]]
            segmentation_cls_id = self.segmentor.labels_decoder().cls_id_dict[self.cls_dict[bbox_cls_id[idx]]]
            mask[mask != segmentation_cls_id] = 0
            mask[mask == segmentation_cls_id] = 1

            targets[idx].mask = mask

        return targets