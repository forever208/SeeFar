"""
test the performance for the speed estimation model on 8 videos

each video has its own config, change the following function and parameter according to the test video:
    1. within_arrow_area_video_1()
    2. self.road_arrow_pixels =
"""


import cv2
import numpy as np
import math


class SpeedEstimationEvaluator():

    def __init__(self, fps):
        self.speed_record = {}
        self.xy_record = {}
        self.id_being_measured = []
        self.evaluation_results = []

        self.fps = fps
        self.road_arrow_size = 6    # the gt size of the road arrow is 6 meters
        self.road_arrow_pixels = 370    # how many pixel does the arrow occupy in the image [370, 370]


    def test_speed(self, image, bboxes):
        """
        test the performance of speed estimator based on the road arrow prior
        @param bboxes: [[x1, y1, x2, y2, id, speed, motion_vec, motion_direction, "in/out"]
        """

        for (x1, y1, x2, y2, id, speed, motion_vec, motion_dir, within_flow) in bboxes:
            if speed:
                # if the car move into the evaluation area, keep record its speed for subsequent speed calculation
                if self.within_arrow_area_video_3(image, x1, y1, x2, y2):
                    if id not in self.id_being_measured:
                        self.id_being_measured.append(id)
                        self.speed_record[id] = [speed]
                        self.xy_record[id] = [[(x1+x2)/2, (y1+y2)/2]]
                    else:
                        self.speed_record[id].append(speed)
                        self.xy_record[id].append([(x1+x2)/2, (y1+y2)/2])

                # if the car used to occur on the arrow area and now has moved out of this area
                else:
                    if id in self.id_being_measured:

                        # compute the gt speed
                        xy_first = self.xy_record[id][0]
                        xy_last = self.xy_record[id][-1]
                        pixel_displacement = math.sqrt((xy_last[0]-xy_first[0])**2 + (xy_last[1]-xy_first[1])**2)
                        gt_displacement = pixel_displacement * (self.road_arrow_size/self.road_arrow_pixels)    # meters
                        frame_interval = len(self.speed_record[id])
                        gt_speed = gt_displacement / ((frame_interval-1)/self.fps)    # m/s
                        gt_speed = gt_speed * 3.6    # km/h

                        # evaluation results
                        avg_speed = abs(np.mean(self.speed_record[id]))
                        speed_error = abs(gt_speed - avg_speed)
                        error_rate = speed_error / gt_speed
                        self.evaluation_results.append([id, gt_speed, avg_speed, speed_error, error_rate])
                        # print('car_ID: ', id)
                        # print('ground_truth speed: ', gt_speed)
                        # print('estimated speed: ', avg_speed)
                        # print(' ')

                        # remove speed and coordinate records in dictionaries
                        self.id_being_measured.remove(id)
                        del self.speed_record[id]
                        del self.xy_record[id]


    def within_arrow_area_video_1(self, image, x1, y1, x2, y2):
        """
        judge that if the bbox is inside the road arrow area
        """

        # plot the road arrows by red rectangular, make sure the coordinates are correct
        cv2.rectangle(image, (1950, 140), (2320, 1940), color=(4, 19, 186), thickness=2, lineType=cv2.LINE_AA)

        xc, yc = (x1+x2)/2, (y1+y2)/2
        if 1950 <= xc <= 2320:    # the arrow is 370 pixels long
            if 140 <= yc <= 1940:
                ret = True
            else:
                ret = False
        else:
            ret = False

        return ret


    def within_arrow_area_video_2(self, image, x1, y1, x2, y2):
        """
        judge that if the bbox is inside the road arrow area
        """

        # plot the road arrows by red rectangular, make sure the coordinates are correct
        cv2.rectangle(image, (1070, 740), (3050, 1100), color=(4, 19, 186), thickness=2, lineType=cv2.LINE_AA)

        xc, yc = (x1+x2)/2, (y1+y2)/2
        if 740 <= yc <= 1100:    # the arrow is 370 pixels long
            if 1070 <= xc <= 3050:
                ret = True
            else:
                ret = False
        else:
            ret = False

        return ret


    def within_arrow_area_video_3(self, image, x1, y1, x2, y2):
        """
        judge that if the bbox is inside the road arrow area
        """

        # plot the road arrows by red rectangular, make sure the coordinates are correct
        arrow_polygon_corners = np.array([[1250, 200], [1180, 850], [1495, 850], [1530, 200]], np.int32)
        cv2.polylines(image, [arrow_polygon_corners], isClosed=True, color=(4, 19, 186), thickness=2)
        # cv2.rectangle(image, (1070, 740), (3050, 1100), color=(4, 19, 186), thickness=2, lineType=cv2.LINE_AA)

        xc, yc = (x1+x2)/2, (y1+y2)/2
        if 740 <= yc <= 1100:    # the arrow is 370 pixels long
            if 1070 <= xc <= 3050:
                ret = True
            else:
                ret = False
        else:
            ret = False

        return ret











