"""
test the performance for the speed estimation model on our 8 videos
"""


import cv2
import numpy as np
import math


class SpeedEstimationEvaluator():

    def __init__(self, fps, cam_angle, drone_speed, video_num):
        self.speed_record = {}
        self.xy_record = {}
        self.id_being_measured = []
        self.evaluation_results = []

        self.cam_angle = cam_angle
        self.drone_speed = drone_speed
        self.fps = fps
        self.video_number = video_num    # the number of the test video, vary from 1 to 8
        self.road_arrow_size = 6    # the gt size of the road arrow is 6 meters
        self.road_arrow_pixels = 370    # the arrow is 370-pixel long in the video 1,2


    def test_speed(self, image, bboxes):
        """
        test the performance of speed estimator based on the road arrow prior
        @param bboxes: [[x1, y1, x2, y2, id, speed, motion_vec, motion_direction, "in/out"]
        """

        for (x1, y1, x2, y2, id, speed, motion_vec, motion_dir, within_flow) in bboxes:
            if speed:

                # for video 1, 2
                if self.cam_angle == 0:
                    # if the car move into the evaluation area, keep record its speed for subsequent speed calculation
                    if self.within_arrow_area_video_1_to_2(image, x1, y1, x2, y2):
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
                            gt_displacement = pixel_displacement * (self.road_arrow_size/self.road_arrow_pixels)    # m
                            frame_interval = len(self.speed_record[id])
                            if frame_interval == 1:
                                continue
                            gt_speed = gt_displacement / ((frame_interval-1)/self.fps)    # m/s
                            gt_speed = gt_speed * 3.6    # km/h

                            # evaluation results
                            avg_speed = abs(np.mean(self.speed_record[id]))
                            speed_error = abs(gt_speed - avg_speed)
                            error_rate = speed_error / gt_speed
                            self.evaluation_results.append([id, gt_speed, avg_speed, speed_error, error_rate])
                            # print('ID: {}  gt speed: {:.1f}  estimated speed: {:.1f}'.format(id, gt_speed, avg_speed))

                            # remove speed and coordinate records in dictionaries
                            self.id_being_measured.remove(id)
                            del self.speed_record[id]
                            del self.xy_record[id]

                # for video 3, 4, 5, 6, 7, 8
                elif self.cam_angle > 0:
                    # skip these cars, because they are miss-detected for a few frames in the arrow area
                    if self.video_number == 3 and id in [4, 47, 48, 57, 78]:
                        continue
                    #  arrow area [6, 54, 57, 79, 98]
                    elif self.video_number == 5 and id in [6, 22, 53, 79, 99]:
                        continue
                    elif self.video_number == 6 and id in [1, 58, 89, 90]:
                        continue
                    elif self.video_number == 7 and id in [1, 56, 70, 91, 94, 102,]:
                        continue


                    within_arrow_area, A,B,C,D,E,F,G,H = self.within_arrow_area_video_3_to_8(image, x1, y1, x2, y2)
                    # if the car move into the evaluation area, keep record its speed for subsequent speed calculation
                    if within_arrow_area:
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
                            # compute the pixel displacement
                            xy_first = self.xy_record[id][0]
                            xy_last = self.xy_record[id][-1]
                            x_displacement = xy_last[0]-xy_first[0]
                            y_displacement = xy_last[1]-xy_first[1]
                            pixel_displacement = math.sqrt(x_displacement**2 + y_displacement**2)
                            if abs(x_displacement) > abs(y_displacement):
                                if x_displacement < 0:
                                    pixel_displacement = -pixel_displacement
                            else:
                                if y_displacement < 0:
                                    pixel_displacement = -pixel_displacement

                            # compute the dynamic GSD in the arrow area by linear interpolation
                            displacement_yc = (xy_last[1] + xy_first[1]) / 2    # y coordinate of the trajectory centre
                            if A[1] <= displacement_yc <= B[1]:
                                AD = D[0] - A[0]    # pixel length of AD
                                BC = C[0] - B[0]
                                arrow_h_pixels = B[1] - A[1]
                                self.road_arrow_pixels = AD + ((displacement_yc-A[1])/arrow_h_pixels) * (BC-AD)
                            elif E[1] <= displacement_yc <= F[1]:
                                EH = H[0] - E[0]    # pixel length of AD
                                FG = G[0] - F[0]
                                arrow_h_pixels = F[1] - E[1]
                                self.road_arrow_pixels = EH + ((displacement_yc-E[1])/arrow_h_pixels) * (FG-EH)

                            # compute the gt speed
                            gt_displacement = pixel_displacement * (self.road_arrow_size/self.road_arrow_pixels)    # m
                            frame_interval = len(self.speed_record[id])
                            if frame_interval == 1:
                                continue
                            gt_speed = gt_displacement / ((frame_interval-1)/self.fps)    # m/s
                            gt_speed = gt_speed * 3.6 + self.drone_speed    # km/h
                            gt_speed = abs(gt_speed)    # ignore the speed direction when doing the evaluation

                            # evaluation results
                            avg_speed = abs(np.mean(self.speed_record[id]))
                            speed_error = abs(gt_speed - avg_speed)
                            error_rate = speed_error / gt_speed
                            self.evaluation_results.append([id, gt_speed, avg_speed, speed_error, error_rate])
                            if speed_error > 6:
                                print('* ID: {}  gt speed: {:.1f}  estimated speed: {:.1f}'.format(id, gt_speed, avg_speed))
                            elif speed_error > 4:
                                print('ID: {}  gt speed: {:.1f}  estimated speed: {:.1f}'.format(id, gt_speed, avg_speed))
                            # print('ID: {}  gt speed: {:.1f}  estimated speed: {:.1f}'.format(id, gt_speed, avg_speed))

                            # remove speed and coordinate records in dictionaries
                            self.id_being_measured.remove(id)
                            del self.speed_record[id]
                            del self.xy_record[id]
                else:
                    raise Exception("the camera angle is wrong")


    def within_arrow_area_video_1_to_2(self, image, x1, y1, x2, y2):
        """
        judge that if the bbox is inside the road arrow area, parameter only applicable for our video 1
        """
        if self.video_number == 1:
            A, D = (1950, 140), (2320, 1940)
        else:
            A, D = (1070, 740), (3050, 1100)

        # plot the road arrows by red rectangular, make sure the coordinates are correct
        cv2.rectangle(image, A, D, color=(4, 19, 186), thickness=2, lineType=cv2.LINE_AA)

        xc, yc = (x1+x2)/2, (y1+y2)/2
        if A[0] <= xc <= D[0]:    # the arrow is 370 pixels long
            if A[1] <= yc <= D[1]:
                ret = True
            else:
                ret = False
        else:
            ret = False

        return ret


    def within_arrow_area_video_3_to_8(self, image, x1, y1, x2, y2):
        """
        judge that if the bbox is inside the road arrow area, parameter only applicable for our video 3
        @param x1, x2, y1, y2: the coordinates of the bbox
        """

        if self.video_number == 3:
            A, B, C, D = [1250, 200], [1180, 850], [1495, 850], [1530, 200]    # 4 corners of the arrow area
            E, F, G, H = [2505, 1050], [2570, 1937], [2955, 1937], [2840, 1050]    # 4 corners of the arrow area
        elif self.video_number == 5:
            A, B, C, D = [1362, 577], [1307, 1175], [1618, 1180], [1654, 582]
            E, F, G, H = [2018, 1330], [2019, 2109], [2379, 2110], [2341, 1335]
        elif self.video_number == 6:
            A, B, C, D = [1086, 76], [1046, 827], [1420, 826], [1439, 72]
            E, F, G, H = [1947, 1004], [1982, 1914], [2392, 1913], [2335, 1000]
        elif self.video_number == 7:
            A, B, C, D = [1250, 458], [1197, 1032], [1520, 1029], [1525, 454]
            E, F, G, H = [1901, 1171], [1900, 1895], [2280, 1884], [2228, 1167]
        elif self.video_number == 8:
            A, B, C, D = [1198, 492], [1087, 1071], [1448, 1071], [1508, 495]
            E, F, G, H = [1903, 1229], [1876, 2096], [2334, 2102], [2283, 1235]
        else:
            raise Exception("the test video is not defined")

        # plot the road arrows by red rectangular, make sure the coordinates are correct
        arrow_polygon_corners = np.array([A, B, C, D], np.int32)    # 280-->315
        cv2.polylines(image, [arrow_polygon_corners], isClosed=True, color=(4, 19, 186), thickness=2)
        arrow_polygon_corners = np.array([E, F, G, H], np.int32)    # 335-->385
        cv2.polylines(image, [arrow_polygon_corners], isClosed=True, color=(4, 19, 186), thickness=2)

        def cross_product(xa, ya, xb, yb, xm, ym):
            vector_ab = (xb-xa, yb-ya)
            vector_am = (xm-xa, ym-ya)
            return vector_ab[0]*vector_am[1] - vector_ab[1]*vector_am[0]

        xc, yc = (x1+x2)/2, (y1+y2)/2
        ret = False
        # if the bbox is inside the first arrow area (the all 4 cross product must be > 0)
        if cross_product(D[0], D[1], C[0], C[1], xc, yc) > 0:
            if cross_product(C[0], C[1], B[0], B[1], xc, yc) > 0:
                if cross_product(B[0], B[1], A[0], A[1], xc, yc) > 0:
                    if cross_product(A[0], A[1], D[0], D[1], xc, yc) > 0:
                        ret = True
        # if the bbox is inside the second arrow area (the all 4 cross product must be > 0)
        if cross_product(H[0], H[1], G[0], G[1], xc, yc) > 0:
            if cross_product(G[0], G[1], F[0], F[1], xc, yc) > 0:
                if cross_product(F[0], F[1], E[0], E[1], xc, yc) > 0:
                    if cross_product(E[0], E[1], H[0], H[1], xc, yc) > 0:
                        ret = True

        return ret, A, B, C, D, E, F, G, H











