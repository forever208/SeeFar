import torch
import cv2
import numpy as np
from detector import Detector
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
from traffic_analysis.dynamic_area import DynamicArea
from traffic_analysis.speed_estimation import SpeedEstimation
from traffic_analysis.flow_counter import FlowCounter


cfg = get_config()
cfg.merge_from_file("deep_sort/configs/deep_sort.yaml")


class TrafficAnalyst():
    """
    Detector + Tracker + Dynamic area + Speed estimator
    """

    def __init__(self, model, width, height):
        self.detector = Detector(model)
        self.deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                                 max_dist=cfg.DEEPSORT.MAX_DIST,
                                 min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                                 nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP,
                                 max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                                 max_age=cfg.DEEPSORT.MAX_AGE,
                                 n_init=cfg.DEEPSORT.N_INIT,
                                 nn_budget=cfg.DEEPSORT.NN_BUDGET,
                                 use_cuda=True)
        self.dynamic_area = DynamicArea()
        self.flow_counter = FlowCounter(width, height)
        self.speed_estimator = SpeedEstimation()

        self.frameCounter = 0
        self.retDict = {'frame': None, 'list_of_ids': None, 'obj_bboxes': []}


    def plot_bboxes(self, image, bboxes, flow_direct1, flow_direct2, line_thickness=None):
        """
        plot all bbox and count the traffic flow
        @param image: original video frame, 3D array (h, w, 3)
        @param bboxes: [[x1, y1, x2, y2, id, speed, motion_vec, motion_direction, in/out],
        @return:
            image: image that has been added bbox, text and dynamic area, 3D array (h, w, 3)
        """

        font_thick = line_thickness or round(0.002 * (image.shape[0] + image.shape[1]) / 2) + 1  # line/font thickness

        # plot bbox, id, speed on the image
        for (x1, y1, x2, y2, id, speed, motion_vec, motion_dir, within_flow) in bboxes:
            # set different color for each tracked object
            if within_flow == 'in':
                if motion_dir == flow_direct1:
                    color = (150, 147, 10)    # bbox in flow1: green (BGR channel)
                elif motion_dir == flow_direct2:
                    color = (249, 187, 0)    # bbox in flow2: blue (BGR channel)
                else:
                    raise Exception("bbox within flow has different directions with both flows")
            elif within_flow == 'out':
                color = (4, 19, 186)    # bbox out of main flow: red (BGR channel)
            elif within_flow == 'few':
                color = (200, 200, 200)    # too few bbox in current frame: gray (BGR channel)
            elif within_flow == 'init':
                color = (0, 155, 238)    # vehicle flow is being initialised: orange (BGR channel)
            else:
                raise Exception("flow attribute is not defined")


            # draw bbox, id, speed
            c1, c2 = (x1, y1), (x2, y2)
            cv2.rectangle(image, c1, c2, color, thickness=int(font_thick / 1.5), lineType=cv2.LINE_AA)
            tf = max(font_thick - 1, 1)  # font thickness
            cv2.putText(image, text='{}-{}'.format(id, speed), org=(c1[0], c1[1] - 2),
                        fontFace=0, fontScale=font_thick / 6, color=color, thickness=int(tf / 1.2),
                        lineType=cv2.LINE_AA)

        # # plot dynamic area on the image
        # if tl is not None:
        #     polygon = np.concatenate((tl, tr, br, bl), axis=0)
        #     polygon = polygon.reshape((-1, 1, 2))
        #     image = cv2.polylines(image, np.int32([polygon]), isClosed=True, color=(255, 255, 0), thickness=2)

        return image


    def update(self, image):
        """
        YOLOv5 + DeepSORT + Dynamic Area + Speed estimation
        @param image: original video frame, 3D array (h, w, 3)
        @return:
        """

        """1. get detections from YOLOv5"""
        _, bboxes = self.detector.detect(image)  # bbox size is regarding to the original video frame
        bbox_xywh = []
        confs = []
        bboxes_tracked = []

        if len(bboxes):
            # Adapt detections to deep sort input format
            for x1, y1, x2, y2, _, conf in bboxes:
                obj = [int((x1+x2)/2), int((y1+y2)/2), x2-x1, y2-y1]
                bbox_xywh.append(obj)
                confs.append(conf)
            xywhs = torch.Tensor(bbox_xywh)
            confss = torch.Tensor(confs)

            """2. Pass detections to DeepSORT"""
            # get outputs: nested list (num_tracks, 5), each element is [x1, y1, x2, y2, track_id]
            bboxes_tracked = self.deepsort.update(xywhs, confss, image)

        """3. Do speed estimation for each car"""
        # bboxes_with_speed: nested list, [[x1, y1, x2, y2, id, speed, motion_vec, motion_direction], []...[]]
        bboxes_with_speed = self.speed_estimator.speed_update(bboxes_tracked, image)

        """4. Do flow detection and flow counter (detect the main road, count the traffic flow)"""
        # bbox_within_flow: nested list, [[x1, y1, x2, y2, id, speed, motion_vec, motion_direction, in/out], []...[]]
        bbox_within_flow = self.flow_counter.main_road_judge(bboxes_with_speed)
        image = self.flow_counter.flow_counter(image, bbox_within_flow)

        # plot bbox, id, dynamic area, speed on the image
        flow_direct1, flow_direct2 = self.flow_counter.flow_direct1, self.flow_counter.flow_direct2
        image = self.plot_bboxes(image, bbox_within_flow, flow_direct1, flow_direct2)

        # return management
        self.frameCounter += 1
        self.retDict['frame'] = image    # image that has been added all info, 3D array (h, w, 3)
        self.retDict['obj_bboxes'] = bbox_within_flow    # [[x1, y1, x2, y2, id, speed, m_vec, m_dir, in/out], []...[]]

        # test
        return self.retDict
