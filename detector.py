import torch
import numpy as np
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.datasets import letterbox
from utils.torch_utils import select_device


class Detector():
    """
    API of YOLOv5 model
    """
    def __init__(self, model):
        self.img_size = 640
        self.threshold = 0.3
        self.stride = 1
        self.init_model(model)
        self.obj_list = ['car']    # obj class that only shown in this project, e.g. ['person', 'car', 'bus', 'truck']


    def init_model(self, model):
        """
        load model and weights of YOLOv5
        """
        self.weights = 'weights/' + model + '.pt'
        self.device = '0' if torch.cuda.is_available() else 'cpu'
        self.device = select_device(self.device)
        model = attempt_load(self.weights, map_location=self.device)
        model.to(self.device).eval()
        model.half()    # only for GPU
        model.float()
        self.m = model
        self.names = model.module.names if hasattr(model, 'module') else model.names


    def preprocess(self, img):
        """
        resize, normalise the original image
        @param img: original video frame, 3D array (h, w, 3)
        @return:
            img0: original video frame
            img: preprocessed image, 4D tensor (batch, 3, h', w')
        """
        img0 = img.copy()
        img = letterbox(img, new_shape=self.img_size)[0]    # resize to (640, h) and h meet the 32 stride demand
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half()    # only for GPU
        img = img.float()
        img /= 255.0    # shrink to [0, 1]
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        return img0, img


    def detect(self, im):
        """
        get detections from YOLOv5, do NMS
        @param im: original video frame, 3D array (h, w, 3)
        @return:
            im: original video frame, 3D array (h, w, 3)
            pred_boxes: 1D tuple list [(x1, y1, x2, y2, cls_name, conf), ()...] regarding to original image size
        """

        # resize, normalise the original image
        im0, img = self.preprocess(im)

        # get detections from model
        pred = self.m(img, augment=False)[0]
        pred = pred.float()

        # do NMS
        pred = non_max_suppression(pred, self.threshold, 0.4)

        # transform bbox coordinates to original image size
        pred_boxes = []
        for det in pred:    # len(pred) = 1
            if det is not None and len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                for *x, conf, cls_id in det:
                    lbl = self.names[int(cls_id)]
                    if not lbl in self.obj_list:
                        continue
                    x1, y1 = int(x[0]), int(x[1])
                    x2, y2 = int(x[2]), int(x[3])

                    # remove bbox near the boarder(edge) to reduce ID switch
                    org_h, org_w = im.shape[0], im.shape[1]
                    if x1<0.02*org_w or x2>0.98*org_w:
                        continue
                    if y1<0.02*org_h or y2>0.98*org_h:
                        continue

                    pred_boxes.append((x1, y1, x2, y2, lbl, conf))

        return im, pred_boxes
