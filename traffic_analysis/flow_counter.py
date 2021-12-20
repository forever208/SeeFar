import numpy as np
import scipy.linalg
import cv2
import math


class FlowCounter():

    def __init__(self, width, height, num_frames=20):
        self.history_frame_ls = []
        self.history_bbox_ls = []
        self.bbox_flow2_backup = []
        self.frames_counter = 0
        self.num_frames = num_frames  # number of frames the history flow direction is preserved
        self.flow_direct1 = None    # first flow direction of history bbox, 'right'/'left'/'down'/'up'/'static'
        self.flow_direct2 = None    # second flow direction of history bbox, 'right'/'left'/'down'/'up'/'static'
        self.flow_vec1 = None    # first flow vector of history bbox, (x_mean, y_mean)
        self.flow_vec2 = None    # second flow vector of history bbox, (x_mean, y_mean)

        self.img_w = width
        self.img_h = height
        self.up_counter = 0
        self.down_counter = 0
        self.right_counter = 0
        self.left_counter = 0
        self.counter_init = False
        self.blue_polygon_history = []    # history track_ID that hit the blue polygon
        self.yellow_polygon_history = []    # history track_ID that hit the yellow polygon
        self.polygon_blue_yellow = None
        self.color_polygons_image = None


    def main_road_judge(self, image, bbox_current_frame):
        """
        1. define the two main flow directions
        2. judge that each bbox is in/out of the main flow via Mahalanobis distance
        3. append 'in/out/init/few' on the tail of each bbox
        @param bbox_current_frame: [[x1, y1, x2, y2, id, speed, motion_vec, motion_direction], []...[]]
        @return:
            bbox_current_frame: [[x1, y1, x2, y2, id, speed, motion_vec, motion_direction, in/out/init/few], []...[]]
        """

        self.frames_counter += 1
        if len(bbox_current_frame) >= 4:
            if len(self.history_frame_ls) == self.num_frames:

                """define the two main flow directions and split history_bbox into 2 groups by the flow directions"""
                image, bbox_flow1, bbox_flow2 = self.flow_direct(image, self.history_bbox_ls)

                """judge that if bboxes are in/out of the main flow via Mahalanobis distance"""
                bbox_current_frame = self.maha_distance(bbox_flow1, bbox_flow2, bbox_current_frame)
            else:
                for cur_bbox in bbox_current_frame:
                    cur_bbox.append('init')
                image = self.plot_sampling(image)

            """historical bbox management (sparse sampling) """
            if len(self.history_frame_ls) == self.num_frames and self.frames_counter % 10 == 0:
                del self.history_frame_ls[0]
            # add current frame into history frame for every 10 frames
            if self.frames_counter % 10 == 0:
                self.history_frame_ls.append(bbox_current_frame)

                # gather historical bbox from previous 10 frames
                if len(self.history_frame_ls) == self.num_frames:
                    self.history_bbox_ls = []
                    for i in range(0, self.num_frames):
                        for his_bbox in self.history_frame_ls[i]:
                            if his_bbox[8] == 'in' or his_bbox[8] == 'init':
                                self.history_bbox_ls.append(his_bbox)

        # mark as 'few' when there is less than 4 bbox in current frame
        else:
            for cur_bbox in bbox_current_frame:
                cur_bbox.append('few')

        return image, bbox_current_frame


    def flow_counter(self, image, bbox_current_frame):
        """
        count the traffic flow volume
        @param image: original image, 3D array
        @param bbox_current_frame: [[x1, y1, x2, y2, id, speed, motion_vec, motion_direction, in/out], []...[]]
        @return:
            image: image that has been added flow counter info and line-hitting area
        """

        if not self.counter_init and self.flow_direct1:
            self.fill_polygon()

        if len(bbox_current_frame) > 0 and self.counter_init == True:
            for (x1, y1, x2, y2, track_id, speed, motion_vec, motion_dir, within_flow) in bbox_current_frame:
                # if current track is in the blue polygon
                if self.polygon_blue_yellow[int((y1+y2)/2), int((x1+x2)/2)] == 1:
                    if track_id not in self.blue_polygon_history:
                        self.blue_polygon_history.append(track_id)
                    # if current track was in yellow polygon before, remark the track as an UP vehicle
                    if track_id in self.yellow_polygon_history:
                        if self.flow_direct1 == 'up' or self.flow_direct1 == 'down':
                            self.up_counter += 1
                        else:
                            self.left_counter += 1
                        # remove the track record in yellow polygon to avoid duplicate count
                        self.yellow_polygon_history.remove(track_id)

                # if current bbox is in the yellow polygon
                elif self.polygon_blue_yellow[int((y1+y2)/2), int((x1+x2)/2)] == 2:
                    if track_id not in self.yellow_polygon_history:
                        self.yellow_polygon_history.append(track_id)
                    # if current track was in blue polygon before, remark the track as a DOWN object
                    if track_id in self.blue_polygon_history:
                        if self.flow_direct1 == 'up' or self.flow_direct1 == 'down':
                            self.down_counter += 1
                        else:
                            self.right_counter += 1
                        # remove the track record in blue polygon to avoid duplicate count
                        self.blue_polygon_history.remove(track_id)

            #     # remove history track_ID that is not in current frame
            #     all_polygon_history = self.blue_polygon_history + self.yellow_polygon_history
            #     for his_id in all_polygon_history:
            #         IS_FOUND = False
            #         for (_, _, _, _, bbox_id, _, _, _, _) in bbox_current_frame:
            #             if bbox_id == his_id:
            #                 IS_FOUND = True
            #             if not IS_FOUND:
            #                 if his_id in self.yellow_polygon_history:
            #                     self.yellow_polygon_history.remove(his_id)
            #                 if his_id in self.blue_polygon_history:
            #                     self.blue_polygon_history.remove(his_id)
            #     all_polygon_history.clear()
            #
            # # clear the overlap history list if no tracks in current frame
            # else:
            #     self.blue_polygon_history.clear()
            #     self.yellow_polygon_history.clear()

            # return the image with flow counter info and historical bbox info
            image = cv2.add(image, self.color_polygons_image)

            if self.flow_direct1 == 'up' or self.flow_direct1 == 'down':
                image = cv2.putText(img=image,
                                    text='DOWN: ' + str(self.down_counter) + ' , UP: ' + str(self.up_counter),
                                    org=(int(self.img_w * 0.1), int(self.img_h * 0.05)),
                                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=self.img_w/2000, color=(255, 255, 255), thickness=int(self.img_w/500))
            else:
                image = cv2.putText(img=image,
                                    text='LEFT: ' + str(self.left_counter) + ' , RIGHT: ' + str(self.right_counter),
                                    org=(int(self.img_w * 0.1), int(self.img_h * 0.05)),
                                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=self.img_w/2000, color=(255, 255, 255), thickness=int(self.img_w/500))

            image = cv2.putText(img=image,
                                text='Flow1: ' + str(self.flow_direct1),
                                org=(int(self.img_w * 0.4), int(self.img_h * 0.05)),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=self.img_w/2000, color=(27, 227, 94), thickness=int(self.img_w/500))

            image = cv2.putText(img=image,
                                text='Flow2: ' + str(self.flow_direct2),
                                org=(int(self.img_w * 0.6), int(self.img_h * 0.05)),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=self.img_w/2000, color=(249, 187, 0), thickness=int(self.img_w/500))

        return image


    def flow_direct(self, image, history_bbox):
        """
        given the previous 19 frames bbox:
            1. find the 2 flow directions, assign the directions to the class attributes
            2. split history_bbox into 2 groups based on the flow directions
            3. compute the mean motion vectors for 2 flows
        @param history_bbox: nested list, [[x1, y1, x2, y2, id, speed, motion_vec, motion_direction], []...[]]
        @return:
            bbox_flow1: the bbox of the first main flow, [[xc, yc], []...[]]
            bbox_flow2: the bbox of the second main flow, [[xc, yc], []...[]], might also be empty list []
        """

        # 1. find the 2 main flow directions ('right'/'left'/'down'/'up'/'static'/'none')
        bbox_direction_ls = []
        for bbox in history_bbox:    # get the direction list for all history bbox
            if bbox[7]:
                bbox_direction_ls.append(bbox[7])

        self.flow_direct1 = max(bbox_direction_ls, key=bbox_direction_ls.count)
        while self.flow_direct1 in bbox_direction_ls:
            bbox_direction_ls.remove(self.flow_direct1)
        self.flow_direct2 = max(bbox_direction_ls, key=bbox_direction_ls.count) if bbox_direction_ls else None

        # 2. split history_bbox into 2 groups based on the flow directions
        bbox_flow1, bbox_flow2 = [], []
        motion_vec_flow1, motion_vec_flow2 = [], []
        for bbox in history_bbox:
            if bbox[7]:
                # valid historical bbox must have the same direction with the 2 main flows
                if bbox[7] == self.flow_direct1:
                    bbox_flow1.append([(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2])
                    motion_vec_flow1.append(bbox[6])
                elif self.flow_direct2 and bbox[7] == self.flow_direct2:
                    bbox_flow2.append([(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2])
                    motion_vec_flow2.append(bbox[6])

        # if too few historical bbox in flow2, use the backup bbox to compute Maha distanace
        if len(bbox_flow2) >= 2 * self.num_frames:
            self.bbox_flow2_backup = bbox_flow2
        if len(bbox_flow2) < 2 * self.num_frames and self.bbox_flow2_backup:
            bbox_flow2 = self.bbox_flow2_backup

        # 3. calculate the mean motion vector for the 2 main flows
        x1_mean, y1_mean = np.array(motion_vec_flow1).mean(axis=0)
        x2_mean, y2_mean = np.array(motion_vec_flow2).mean(axis=0) if motion_vec_flow2 else (None, None)
        self.flow_vec1 = (x1_mean, y1_mean)
        self.flow_vec2 = (x2_mean, y2_mean)

        # 4. plot the historical bbox as a scatter for the 2 main flows
        for bbox in bbox_flow1:
            image = cv2.circle(img=image,
                               center=(int(bbox[0]), int(bbox[1])), radius=int(self.img_w/600),
                               color=(27, 227, 94), thickness=int(self.img_w/1000))
        for bbox in bbox_flow2:
            image = cv2.circle(img=image,
                               center=(int(bbox[0]), int(bbox[1])), radius=int(self.img_w/600),
                               color=(249, 187, 0), thickness=int(self.img_w/1000))

        return image, bbox_flow1, bbox_flow2


    def maha_distance(self, bbox_flow1, bbox_flow2, bbox_current_frame):
        """
        compute the mahalanobis distance between each bbox and its flow centre derived from previous 19 frames
        @param bbox_flow1: history bbox within the first flow, nested list [[xc, yc], []...[]]
        @param bbox_flow2: history bbox within the second flow,,nested list [[xc, yc], []...[]]
        @param bbox_current_frame: nested list, [[x1, y1, x2, y2, id, speed, motion_vec, motion_direction], []...[]]
        @return:
            bbox_current_frame: nested list, [[x1, y1, x2, y2, id, speed, motion_vec, motion_direction, in/out], []...[]]
        """

        # compute mean and covariance
        flow1_array, flow2_array = np.array(bbox_flow1, dtype=float), np.array(bbox_flow2, dtype=float)
        flow2_num_samples = flow2_array.T.shape[1] if flow2_array.any() else 0

        # if the number of samples in flow2 is too little, don't compute Maha distance for the flow2
        if flow2_num_samples >= self.num_frames - 1:
            mean1, mean2 = np.mean(flow1_array.T, axis=1), np.mean(flow2_array.T, axis=1)
            cov1, cov2 = np.cov(flow1_array.T), np.cov(flow2_array.T)    # cov = L * L.trans
            L1, L2 = np.linalg.cholesky(cov1), np.linalg.cholesky(cov2)
        else:
            mean1, mean2 = np.mean(flow1_array.T, axis=1), None
            cov1, cov2 = np.cov(flow1_array.T), None    # cov = L * L.trans
            L1, L2 = np.linalg.cholesky(cov1), None

        # compute Mahala distance between each current bbox and history flow centre, then assign 'in/out/init'
        for bbox in bbox_current_frame:
            if len(bbox) == 8:
                # if bbox has the same motion direction with flow 1
                if bbox[7] == self.flow_direct1:
                    squared_maha = self.maha_calculator(bbox[0], bbox[1], bbox[2], bbox[3], mean1, L1)

                # if bbox has the same motion direction with flow 2
                elif self.flow_direct2 and bbox[7] == self.flow_direct2:
                    squared_maha = self.maha_calculator(bbox[0], bbox[1], bbox[2], bbox[3], mean2, L2) if L2 is not None else 1000

                # the motion direction (up/down/left/right) is inaccurate when car is too far or move along the diagonal
                # Therefore, we need to further compute the angle between motion vector and flow motion vector
                elif bbox[7]:
                    theta1 = self.angle_of_vectors(bbox[6], self.flow_vec1)
                    theta2 = self.angle_of_vectors(bbox[6], self.flow_vec2) if L2 is not None else 180
                    if bbox[7] == 'static':
                        squared_maha = 1000
                    # if the angle between motion vector and the flow_1 < 45 degree, compute the maha distance
                    elif theta1 < theta2 and theta1 < 45:
                        squared_maha = self.maha_calculator(bbox[0], bbox[1], bbox[2], bbox[3], mean1, L1)
                        bbox[7] = self.flow_direct1
                    # if the angle between motion vector and the flow_2 < 45 degree, compute the maha distance
                    elif theta2 < theta1 and theta2 < 45:
                        squared_maha = self.maha_calculator(bbox[0], bbox[1], bbox[2], bbox[3], mean2, L2)
                        bbox[7] = self.flow_direct2
                    # different motion direction
                    else:
                        squared_maha = 1000

                # if the bbox is new, it has no motion info, therefore mark as 'init'
                else:
                    squared_maha = 2000

                # assign in/out according to the Maha distance
                if squared_maha <= 5.99*2:    # 95% confidence interval belongs to the chi-squared distribution
                    bbox.append('in')
                elif squared_maha == 2000:
                    bbox.append('init')
                else:
                    bbox.append('out')
                    # print('outlier distance: ', squared_maha)

        return bbox_current_frame


    def maha_calculator(self, x1, y1, x2, y2, mean, L_matrix):
        """
        compute squared mahalanobis distance
        """
        xc = (x1 + x2) / 2
        yc = (y1 + y2) / 2
        delta = np.array([xc, yc]) - mean
        z = scipy.linalg.solve_triangular(L_matrix, delta.T, lower=True, check_finite=False, overwrite_b=True)
        squared_maha = np.sum(z * z, axis=0)

        return squared_maha


    def angle_of_vectors(self, v1, v2):
        """
        compute the cos angle between 2 vectors
        @param v1: (x1, y1)
        @param v2: (x2, y2)
        @return:
            theta: degree between (0, 180)
        """
        vector_product = v1[0] * v2[0] + v1[1] * v2[1]
        vector_magnitude = math.sqrt(v1[0]**2 + v1[1]**2) * math.sqrt(v2[0]**2 + v2[1]**2)
        cos = vector_product * 1.0 / (vector_magnitude * 1.0 + 1e-6)
        theta = (math.acos(cos) / math.pi) * 180

        return theta


    def fill_polygon(self):
        """
        fill in the the polygon of the vehicle counter area
        """

        # configure the 4 corners points of the blue and yellow polygon
        if self.flow_direct1 == 'up' or self.flow_direct1 == 'down':
            blue_polygon_corners = np.array([[0, self.img_h*0.49], [0, self.img_h*0.51],
                                             [self.img_w-1, self.img_h*0.51], [self.img_w-1, self.img_h*0.49]],
                                            np.int32)
            yellow_polygon_corners = np.array([[0, self.img_h*0.51+1], [0, self.img_h*0.53],
                                               [self.img_w-1, self.img_h*0.53], [self.img_w-1, self.img_h*0.51+1]],
                                              np.int32)
        else:
            blue_polygon_corners = np.array([[self.img_w*0.495, 0], [self.img_w*0.495, self.img_h-1],
                                             [self.img_w*0.51, self.img_h-1], [self.img_w*0.51, 0]],
                                            np.int32)
            yellow_polygon_corners = np.array([[self.img_w*0.51+1, 0], [self.img_w*0.51+1, self.img_h-1],
                                               [self.img_w*0.525, self.img_h-1], [self.img_w*0.525, 0]],
                                              np.int32)

        # fill in the blue polygon (assign 1 within the polygon based on the zero array)
        mask_img = np.zeros((self.img_h, self.img_w), dtype=np.uint8)
        blue_polygon = cv2.fillPoly(mask_img, [blue_polygon_corners], color=1)
        blue_polygon = blue_polygon[:, :, np.newaxis]    # 3D array (h, w, 1)

        # fill in the yellow polygon (assign 2 within the polygon based on the zero array)
        mask_img = np.zeros((self.img_h, self.img_w), dtype=np.uint8)
        yellow_polygon = cv2.fillPoly(mask_img, [yellow_polygon_corners], color=2)
        yellow_polygon = yellow_polygon[:, :, np.newaxis]    # 3D array (h, w, 1)

        # mask array for line-hitting judgement，including 2 polygons (element is either 0/1/2)
        self.polygon_blue_yellow = blue_polygon + yellow_polygon    # 1 stands for blue, 2 stands for yellow
        self.polygon_blue_yellow = cv2.resize(self.polygon_blue_yellow, (self.img_w, self.img_h))

        # polygon array --> polygon image
        blue_color_plate = [255, 0, 0]
        blue_image = np.array(blue_polygon * blue_color_plate, np.uint8)
        yellow_color_plate = [0, 255, 255]
        yellow_image = np.array(yellow_polygon * yellow_color_plate, np.uint8)
        self.color_polygons_image = blue_image + yellow_image
        self.counter_init = True


    def plot_sampling(self, image):
        """
        plot 'vehicle flow is under sampling' on the image at the beginning of the model
        """
        image = cv2.putText(img=image,
                            text='vehicle flow is under sampling',
                            org=(int(self.img_w * 0.3), int(self.img_h * 0.5)),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=self.img_w / 1000, color=(0, 155, 238),
                            thickness=int(self.img_h / 300))

        return image