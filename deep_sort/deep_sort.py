import numpy as np

from .deep.feature_extractor import Extractor, ExtractorVehicle
from .sort.nn_matching import NearestNeighborDistanceMetric
from .sort.preprocessing import non_max_suppression
from .sort.detection import Detection
from .sort.tracker import Tracker
import torch


__all__ = ['DeepSort']


class DeepSort(object):
    def __init__(self, model_path, vehicle_model_path, max_dist=0.2, min_confidence=0.3, nms_max_overlap=1.0, max_iou_distance=0.7, max_age=70, n_init=3, nn_budget=100, use_cuda=True):
        self.min_confidence = min_confidence
        self.nms_max_overlap = nms_max_overlap

        self.extractor = Extractor(model_path, use_cuda=use_cuda)
        self.vehicle_extractor = ExtractorVehicle(vehicle_model_path, use_cuda=use_cuda)
        max_cosine_distance = max_dist
        # nn_budget = 100
        metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = Tracker(metric, max_iou_distance=max_iou_distance, max_age=max_age, n_init=n_init)

    def update(self, bbox_xywh, confidences, cls_ids, ori_img):
        self.height, self.width = ori_img.shape[:2]
        # generate detections
        features = self._get_features(bbox_xywh, ori_img, cls_ids)   # get recognition features for every bbox
        bbox_tlwh = self._xywh_to_tlwh(bbox_xywh)
        detections = [Detection(bbox_tlwh[i], conf, cls_ids[i], features[i]) for i, conf in enumerate(confidences) if conf>self.min_confidence]

        # run on non-maximum supression
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = non_max_suppression(boxes, self.nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # update tracker
        self.tracker.predict()  # predicting bbox position based on kf
        self.tracker.update(detections) # matching bbox to known tracks / creating new tracks

        # output bbox identities
        output_tracks = []
        detections_conf = []
        detections_cls_id = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            box = track.to_tlwh()
            x1,y1,x2,y2 = self._tlwh_to_xyxy(box)
            track_id = track.track_id
            detections_conf.append(track.confidence)
            detections_cls_id.append(track.cls_id)
            # output_tracks.append(np.array([x1,y1,x2,y2, track_id], dtype=np.int))
            output_tracks.append(track)

        # if len(output_tracks) > 0:
        #     output_tracks = np.stack(output_tracks,axis=0)

        return output_tracks, detections#, detections_conf, detections_cls_id

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
            person_features = self.extractor(im_crops)
            # making person and vechiles orthogonal for cosine similarity
            person_features = np.hstack((person_features, np.zeros_like(person_features)))
        else:
            person_features = np.array([])

        if im_crops_vehicles:
            vehicle_features = self.vehicle_extractor(im_crops_vehicles)
            # making person and vechiles orthogonal for cosine similarity
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


