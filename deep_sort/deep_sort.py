import numpy as np
from detector import build_detector
# from . import build_tracker
from .deep.feature_extractor import Extractor, ExtractorVehicle
from .sort.nn_matching import NearestNeighborDistanceMetric
from .sort.preprocessing import non_max_suppression
from .sort.detection import Detection
from .sort.tracker import Tracker
import torch


__all__ = ['DeepSort']


class DeepSort(object):
    def __init__(self, cfg, use_cuda=True):
        self.min_confidence = cfg.DEEPSORT.MIN_CONFIDENCE
        self.nms_max_overlap = cfg.DEEPSORT.NMS_MAX_OVERLAP

        self.detector = build_detector(cfg, use_cuda=use_cuda)
        # self.tracker = build_tracker(cfg, use_cuda=use_cuda)

        self.person_extractor = Extractor(cfg.DEEPSORT.PERSON_REID_CKPT, use_cuda=use_cuda)
        self.vehicle_extractor = ExtractorVehicle(cfg.DEEPSORT.VEHICLE_REID_CKPT, use_cuda=use_cuda)
        max_cosine_distance = cfg.DEEPSORT.MAX_DIST
        metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance, cfg.DEEPSORT.NN_BUDGET)
        self.tracker = Tracker(metric, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                               max_age=cfg.DEEPSORT.MAX_AGE,
                               n_init=cfg.DEEPSORT.N_INIT,
                               max_uncertainty_radius=cfg.DEEPSORT.MAX_KF_UNCERTAINTY_RADIUS)

        k, v = zip(*list(cfg.CLS_DICT.items())[:-2])
        k = [int(x) for x in k]
        self.cls_dict = dict(zip(k, v))

    def detect(self, ori_img):

        # do detection
        bbox_xywh, cls_conf, cls_ids = self.detector(ori_img)  # get all detections from image

        detections = []
        # select supported classes
        mask = np.isin(cls_ids[0], list(self.cls_dict.keys()))
        bbox_xywh = bbox_xywh[0][mask]
        confidences = cls_conf[0][mask]
        cls_ids = cls_ids[0][mask]

        # all cls_ids representing the same cls, will get the same cls_id:
        for i, cls_id in enumerate(cls_ids):
            cls_ids[i] = np.min([k for k,v in self.cls_dict.items() if v == self.cls_dict[cls_id]])

        self.height, self.width = ori_img.shape[:2]

        # generate detections
        features = self._get_features(bbox_xywh, ori_img, cls_ids)  # get recognition features for every bbox
        bbox_tlwh = self._xywh_to_tlwh(bbox_xywh)
        detections = [Detection(bbox_tlwh[i], conf, cls_ids[i], features[i]) for i, conf in enumerate(confidences) if
                      conf > self.min_confidence]

        # run on non-maximum supression
        boxes = np.empty((0, 4))
        confs = np.empty((0, 1))
        cls_ids = np.empty((0, 1))

        for d in detections:
            boxes = np.vstack((boxes, np.array(d.tlwh)))
            confs = np.vstack((confs, np.array([d.confidence])))
            cls_ids = np.vstack((cls_ids, np.array([d.cls_id])))

        confs = confs.squeeze(1)
        cls_ids = cls_ids.squeeze(1)

        # suppressing detections that are too close
        indices = non_max_suppression(boxes.copy(), self.nms_max_overlap, scores=confs, cls_ids=cls_ids, img_shape=ori_img.shape)
        detections = [detections[i] for i in indices]

        return detections

    def track(self, detections, cam2world):

        # update tracker
        self.tracker.predict(cam2world)  # predicting bbox position based on kf
        self.tracker.update(detections, cam2world) # matching bbox to known tracks / creating new tracks

        # output bbox identities
        output_tracks = []
        for track in self.tracker.tracks:
            if not track.is_confirmed():# or track.time_since_update > 0:
                continue
            output_tracks.append(track)

        return output_tracks

    """
    TODO:
        Convert bbox from xc_yc_w_h to xtl_ytl_w_h
    Thanks JieChen91@github.com for reporting this bug!
    """
    @staticmethod
    def _xywh_to_tlwh(bbox_xywh):
        if isinstance(bbox_xywh, np.ndarray):
            bbox_tlwh = bbox_xywh.copy()
        elif isinstance(bbox_xywh, torch.Tensor):
            bbox_tlwh = bbox_xywh.clone()
        bbox_tlwh[:,0] = bbox_xywh[:,0] - bbox_xywh[:,2]/2.
        bbox_tlwh[:,1] = bbox_xywh[:,1] - bbox_xywh[:,3]/2.
        return bbox_tlwh


    def _xywh_to_xyxy(self, bbox_xywh):
        x,y,w,h = bbox_xywh
        x1 = max(int(x-w/2),0)
        x2 = min(int(x+w/2),self.width-1)
        y1 = max(int(y-h/2),0)
        y2 = min(int(y+h/2),self.height-1)
        return x1,y1,x2,y2

    def _tlwh_to_xyxy(self, bbox_tlwh):
        """
        TODO:
            Convert bbox from xtl_ytl_w_h to xc_yc_w_h
        Thanks JieChen91@github.com for reporting this bug!
        """
        x,y,w,h = bbox_tlwh
        x1 = max(int(x),0)
        x2 = min(int(x+w),self.width-1)
        y1 = max(int(y),0)
        y2 = min(int(y+h),self.height-1)
        return x1,y1,x2,y2
    
    def _get_features(self, bbox_xywh, ori_img, cls_ids):
        im_crops = []
        im_crops_vehicles = []
        person_inds = []
        vehicle_inds = []
        for ind, (box, cls_id) in enumerate(zip(bbox_xywh, cls_ids)):
            x1,y1,x2,y2 = self._xywh_to_xyxy(box)
            im = ori_img[y1:y2,x1:x2]
            if cls_id == 0:
                person_inds.append(ind)
                im_crops.append(im)
            else:
                vehicle_inds.append(ind)
                im_crops_vehicles.append(im)

        if im_crops:
            person_features = self.person_extractor(im_crops)
            # making person and vechiles orthogonal for cosine similarity
            person_features = np.hstack((person_features, np.zeros_like(person_features)))
        else:
            person_features = np.array([])

        if im_crops_vehicles:
            vehicle_features = self.vehicle_extractor(im_crops_vehicles)
            # making person and vehicles orthogonal for cosine similarity
            vehicle_features = np.hstack((np.zeros_like(vehicle_features), vehicle_features))
        else:
            vehicle_features = np.array([])

        # concatenating features
        features = np.zeros((len(person_inds) + len(vehicle_inds), max(np.shape(person_features)[-1], np.shape(vehicle_features)[-1])))
        if person_inds:
            features[person_inds, :] = person_features

        if vehicle_inds:
            features[vehicle_inds, :] = vehicle_features

        return features


