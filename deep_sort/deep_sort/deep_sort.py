"""
Entrance of DeepSORT algorithm
"""

import numpy as np
import torch

from .deep.feature_extractor import Extractor
from .sort.nn_matching import NearestNeighborDistanceMetric
from .sort.preprocessing import non_max_suppression
from .sort.detection import Detection
from .sort.tracker import Tracker


__all__ = ['DeepSort']    # __all__ 提供了暴露接口用的”白名单“


class DeepSort(object):
    def __init__(self, model_path, max_dist=0.2, min_confidence=0.3, nms_max_overlap=1.0, max_iou_distance=0.7,
                 max_age=70, n_init=3, nn_budget=100, use_cuda=True):
        self.min_confidence = min_confidence    # 0.3 as default
        self.nms_max_overlap = nms_max_overlap    # 1 stands for no NMS

        # CNN instance, extract the features of a bbox
        self.extractor = Extractor(model_path, use_cuda=use_cuda)

        max_cosine_distance = max_dist    # default as 0.2, threshold of the cosine distance between appearance features
        nn_budget = nn_budget    # max number of feature history, delete history that exceed 100

        # key class instance, find the closest detection for each track (cosine/euclidean distance)
        metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = Tracker(metric, max_iou_distance=max_iou_distance, max_age=max_age, n_init=n_init)


    def update(self, bbox_xywh, confidences, ori_img):
        """
        do NMS for detections,
        @param bbox_xywh: 2D tensor (num_detections, 4), each row is bbox coordinate [x,y,w,h] (original image size)
        @param confidences: 1D tensor (num_detections), confidence of each detection
        @param ori_img: 3D array (h, w, 3)
        @return:
            outputs: 2D array (num_tracks, 5), each row is [x1, y1, x2, y2, track_id]
        """

        self.height, self.width = ori_img.shape[:2]

        # 1. crop bbox from original image, get the corresponding feature embeddings
        features = self._get_features(bbox_xywh, ori_img)    # 2D array, (num_detections, 512)
        bbox_tlwh = self._xywh_to_tlwh(bbox_xywh)    # 2D tensor, (num_detections, 512)

        # 2. construct a list of detected bbox [detection, detection...]
        # each detection is a class instance with attributes (Detection.confidence, Detection.feature, Detection.tlwh)
        detections = [Detection(bbox_tlwh[i], conf, features[i]) for i,conf in enumerate(confidences) if conf>self.min_confidence]

        # 3. do NMS for detections
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = non_max_suppression(boxes, self.nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        """ key functions of DeepSORT """
        # 4. Kalman filter predicts track ==> matching cascade ==> IOU matching ==> confirmed tracks
        self.tracker.predict()    # predict the active tracks for next frame by Kalman filter
        self.tracker.update(detections)    # Perform bbox matching and track management

        # 5. output confirmed matched tracks and their IDs
        outputs = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            box = track.to_tlwh()
            x1,y1,x2,y2 = self._tlwh_to_xyxy(box)
            track_id = track.track_id
            outputs.append([int(x1), int(y1), int(x2), int(y2), int(track_id)])

        return outputs


    @staticmethod
    def _xywh_to_tlwh(bbox_xywh):
        """
        convert bbox from [x,y,w,h] to [t,l,w,h]
        """
        if isinstance(bbox_xywh, np.ndarray):
            bbox_tlwh = bbox_xywh.copy()
        elif isinstance(bbox_xywh, torch.Tensor):
            bbox_tlwh = bbox_xywh.clone()
        bbox_tlwh[:,0] = bbox_xywh[:,0] - bbox_xywh[:,2]/2.
        bbox_tlwh[:,1] = bbox_xywh[:,1] - bbox_xywh[:,3]/2.
        return bbox_tlwh


    def _xywh_to_xyxy(self, bbox_xywh):
        """
        convert bbox from [x y w h] to [x1 y1 x2 y2]
        """
        x,y,w,h = bbox_xywh
        x1 = max(int(x-w/2),0)
        x2 = min(int(x+w/2),self.width-1)
        y1 = max(int(y-h/2),0)
        y2 = min(int(y+h/2),self.height-1)
        return x1,y1,x2,y2


    def _tlwh_to_xyxy(self, bbox_tlwh):
        """
        convert bbox from [t,l,w,h] to [x1 y1 x2 y2]
        """
        x,y,w,h = bbox_tlwh
        x1 = max(int(x),0)
        x2 = min(int(x+w),self.width-1)
        y1 = max(int(y),0)
        y2 = min(int(y+h),self.height-1)
        return x1,y1,x2,y2


    def _xyxy_to_tlwh(self, bbox_xyxy):
        """
        convert bbox from [x1 y1 x2 y2] to [t,l,w,h]
        """
        x1,y1,x2,y2 = bbox_xyxy
        t = x1
        l = y1
        w = int(x2-x1)
        h = int(y2-y1)
        return t,l,w,h


    def _get_features(self, bbox_xywh, ori_img):
        """
        extract the features inside a bbox
        @param bbox_xywh: 2D tensor (num_detections, 4), each row is bbox coordinate [x,y,w,h] (original image size)
        @param ori_img: 3D array (h, w, 3)
        @return:
            features: 2D array, (num_detections, 512), each row is the feature embedding of a bbox
        """
        im_crops = []
        for box in bbox_xywh:
            x1,y1,x2,y2 = self._xywh_to_xyxy(box)
            im = ori_img[y1:y2,x1:x2]    # get the cropped image (i.e. image patch of a bbox)
            im_crops.append(im)
        if im_crops:
            features = self.extractor(im_crops)    # get the features of a cropped image
        else:
            features = np.array([])
        return features


