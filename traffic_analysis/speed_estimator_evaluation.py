import cv2
import numpy as np


class SpeedEstimationEvaluator():

    def __init__(self, fps):
        self.speed_record = {}
        self.id_being_measured = []
        self.evaluation_results = []

        self.fps = fps
        self.road_arrow_size = 6    # the gt size of the road arrow is 6 meters


    def plot_road_arrows(self, image):

        # plot the road arrows by red rectangular, make sure the coordinates are correct
        red = (4, 19, 186)
        cv2.rectangle(image, (1950, 140), (2320, 300),
                      color=red, thickness=2, lineType=cv2.LINE_AA)
        cv2.rectangle(image, (1950, 350), (2320, 500),
                      color=red, thickness=2, lineType=cv2.LINE_AA)
        cv2.rectangle(image, (1950, 570), (2320, 710),
                      color=red, thickness=2, lineType=cv2.LINE_AA)
        cv2.rectangle(image, (1950, 780), (2320, 910),
                      color=red, thickness=2, lineType=cv2.LINE_AA)

        cv2.rectangle(image, (1950, 1150), (2320, 1310),
                      color=red, thickness=2, lineType=cv2.LINE_AA)
        cv2.rectangle(image, (1950, 1380), (2320, 1520),
                      color=red, thickness=2, lineType=cv2.LINE_AA)
        cv2.rectangle(image, (1950, 1590), (2320, 1730),
                      color=red, thickness=2, lineType=cv2.LINE_AA)
        cv2.rectangle(image, (1950, 1800), (2320, 1940),
                      color=red, thickness=2, lineType=cv2.LINE_AA)


    def test_speed(self, bboxes):
        """
        test the performance of speed estimator based on the road arrow prior
        @param bboxes: [[x1, y1, x2, y2, id, speed, motion_vec, motion_direction, "in/out"]
        @return:
        """

        for (x1, y1, x2, y2, id, speed, motion_vec, motion_dir, within_flow) in bboxes:

            # if the car move into the evaluation area, keep record its speed for subsequent speed calculation
            if self.within_arrow_area(x1, y1, x2, y2):
                if id not in self.id_being_measured:
                    self.id_being_measured.append(id)
                    self.speed_record[id] = [speed]
                else:
                    self.speed_record[id].append(speed)

            # if the car used to occur on the arrow area and now has moved out of this area
            else:
                if id in self.id_being_measured:
                    if self.speed_record[id][0]:

                        # compute the gt speed
                        frame_interval = len(self.speed_record[id])
                        gt_speed = self.road_arrow_size / (frame_interval/self.fps)    # m/s
                        gt_speed = gt_speed * 3.6    # km/h

                        # final evaluation results
                        avg_speed = abs(np.mean(self.speed_record[id]))
                        speed_error = abs(gt_speed - avg_speed) / 2
                        error_rate = speed_error / gt_speed
                        self.evaluation_results.append([id, gt_speed, avg_speed, speed_error, error_rate])
                        # print('car_ID: ', id)
                        # print('ground_truth speed: ', int(gt_speed))
                        # print('estimated speed: ', avg_speed)
                        # print(' ')

                        # remove speed record in dictionary and id record in list
                        self.id_being_measured.remove(id)
                        del self.speed_record[id]

    def within_arrow_area(self, x1, y1, x2, y2):
        """
        judge that if the bbox is inside the road arrow area
        """

        xc = (x1 + x2) / 2
        yc = (y1 + y2) / 2
        if 1950 <= xc <= 2320:
            if 140 <= yc <= 300:
                ret = True
            elif 350 <= yc <= 500:
                ret = True
            elif 570 <= yc <= 710:
                ret = True
            elif 780 <= yc <= 910:
                ret = True
            elif 1150 <= yc <= 1310:
                ret = True
            elif 1380 <= yc <= 1520:
                ret = True
            elif 1590 <= yc <= 1730:
                ret = True
            elif 1800 <= yc <= 1940:
                ret = True
            else:
                ret = False
        else:
            ret = False

        return ret











