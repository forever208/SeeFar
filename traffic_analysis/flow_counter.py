import numpy as np
import scipy.linalg
import cv2


class FlowCounter():

    def __init__(self, num_frames=20):
        self.bbox_history_ls = []
        self.num_frames = num_frames  # number of frames the dynamic area maintains
        self.flow_direct1 = None    # string, 'right'/'left'/'down'/'up'/'static'
        self.flow_direct2 = None    # string, 'right'/'left'/'down'/'up'/'static'

        self.img_width = 1280
        self.img_height = 720
        self.up_counter = 0
        self.down_counter = 0
        self.counter_init = False
        self.blue_polygon_history = []    # history track_ID that hit the blue polygon
        self.yellow_polygon_history = []    # history track_ID that hit the yellow polygon
        self.polygon_blue_yellow = None
        self.color_polygons_image = None


    def main_road_judge(self, bbox_current_frame):
        """
        1. define the two main flow directions
        2. judge that each bbox is in/out of the main flow via Mahalanobis distance
        @param bbox_current_frame: nested list, [[x1, y1, x2, y2, id, speed, motion_vec, motion_direction], []...[]]
        @return:
            bbox_current_frame: [[x1, y1, x2, y2, id, speed, motion_vec, motion_direction, in/out], []...[]]
        """

        if len(bbox_current_frame) > 2:
            if len(self.bbox_history_ls) == self.num_frames-1:
                # TODO: only use history bbox that are already in the flow to compute Mahala distance
                # get all bbox of previous 19 frames
                bbox_before = []
                for i in range(0, self.num_frames-1):
                    for his_bbox in self.bbox_history_ls[i]:
                        bbox_before.append(his_bbox)

                """define the two main flow directions and split history_bbox into 2 groups by the flow directions"""
                bbox_flow1, bbox_flow2 = self.flow_direct(bbox_before)    # nested list [[xc, yc], []...[]]

                """judge if bboxes are in/out of the main flow via Mahalanobis distance"""
                bbox_current_frame = self.maha_distance(bbox_flow1, bbox_flow2, bbox_current_frame)
                del self.bbox_history_ls[0]
            else:
                for cur_bbox in bbox_current_frame:
                    cur_bbox.append('init')
            self.bbox_history_ls.append(bbox_current_frame)  # add current frame bbox into the bbox history

        # TODO: consider the situation when a frame has 1 or 2 bbox
        else:
            for cur_bbox in bbox_current_frame:
                cur_bbox.append('few')

        return bbox_current_frame


    def flow_counter(self, image, bbox_current_frame):
        """

        @param image: original image, 3D array
        @param bbox_current_frame: [[x1, y1, x2, y2, id, speed, motion_vec, motion_direction, in/out], []...[]]
        @return:
            image: image that has been added flow counter info and line-hitting area
        """

        if not self.counter_init:
            # fill in the blue polygon (assign 1 within the polygon based on the zero array)
            mask_img = np.zeros((self.img_height, self.img_width), dtype=np.uint8)
            blue_polygon_corners = np.array([[0, 550], [0, 560], [1279, 560], [1279, 550]], np.int32)
            blue_polygon = cv2.fillPoly(mask_img, [blue_polygon_corners], color=1)
            blue_polygon = blue_polygon[:, :, np.newaxis]    # 3D array (h, w, 1)

            # fill in the yellow polygon (assign 2 within the polygon based on the zero array)
            mask_img = np.zeros((self.img_height, self.img_width), dtype=np.uint8)
            yellow_polygon_corners = np.array([[0, 561], [0, 571], [1279, 571], [1279, 561]], np.int32)
            yellow_polygon = cv2.fillPoly(mask_img, [yellow_polygon_corners], color=2)
            yellow_polygon = yellow_polygon[:, :, np.newaxis]    # 3D array (h, w, 1)

            # mask array for line-hitting judgement，including 2 polygons (element is either 0/1/2)
            self.polygon_blue_yellow = blue_polygon + yellow_polygon    # 1 stands for blue, 2 stands for yellow
            self.polygon_blue_yellow = cv2.resize(self.polygon_blue_yellow, (self.img_width, self.img_height))

            # polygon array --> polygon image
            blue_color_plate = [255, 0, 0]
            blue_image = np.array(blue_polygon * blue_color_plate, np.uint8)
            yellow_color_plate = [0, 255, 255]
            yellow_image = np.array(yellow_polygon * yellow_color_plate, np.uint8)
            self.color_polygons_image = blue_image + yellow_image
            self.counter_init = True

        if len(bbox_current_frame) > 0:
            for (x1, y1, x2, y2, track_id, speed, motion_vec, motion_dir, within_flow) in bbox_current_frame:
                # if current track is in the blue polygon
                if self.polygon_blue_yellow[int((y1+y2)/2), int((x1+x2)/2)] == 1:
                    if track_id not in self.blue_polygon_history:
                        self.blue_polygon_history.append(track_id)
                    # if current track was in yellow polygon before, remark the track as an UP vehicle
                    if track_id in self.yellow_polygon_history:
                        self.up_counter += 1
                        print('up count:', self.up_counter, ', up id:', self.yellow_polygon_history)
                        # remove the track record in yellow polygon to avoid duplicate count
                        self.yellow_polygon_history.remove(track_id)

                # if current bbox is in the yellow polygon
                elif self.polygon_blue_yellow[int((y1+y2)/2), int((x1+x2)/2)] == 2:
                    if track_id not in self.yellow_polygon_history:
                        self.yellow_polygon_history.append(track_id)
                    # if current track was in blue polygon before, remark the track as a DOWN object
                    if track_id in self.blue_polygon_history:
                        self.down_counter += 1
                        print('down count:', self.down_counter, ', down id:', self.blue_polygon_history)
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

        # return the image with flow counter info
        image = cv2.add(image, self.color_polygons_image)
        text_draw = 'DOWN: ' + str(self.down_counter) + ' , UP: ' + str(self.up_counter)
        image = cv2.putText(img = image,
                            text = text_draw,
                            org = (int(self.img_width * 0.01), int(self.img_height * 0.05)),
                            fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale = 0.75, color = (150, 147, 10), thickness = 2)
        return image


    def flow_direct(self, history_bbox):
        """
        given the previous 19 frames bbox
        1. find the 2 flow directions, assign the directions to the class attributes
        2. split history_bbox into 2 groups based on the flow directions
        @param history_bbox: nested list, [[x1, y1, x2, y2, id, speed, motion_vec, motion_direction], []...[]]
        @return:
            bbox_flow1: the bbox of the first main flow, [[xc, yc], []...[]]
            bbox_flow2: the bbox of the second main flow, [[xc, yc], []...[]]
        """

        # get the direction list for all history bbox
        bbox_direction_ls = []
        for bbox in history_bbox:
            if bbox[7]:
                bbox_direction_ls.append(bbox[7])

        # find the 2 flow directions ('right'/'left'/'down'/'up'/'static')
        self.flow_direct1 = max(bbox_direction_ls, key=bbox_direction_ls.count)
        while self.flow_direct1 in bbox_direction_ls:
            bbox_direction_ls.remove(self.flow_direct1)
        self.flow_direct2 = max(bbox_direction_ls, key=bbox_direction_ls.count)

        # split history_bbox into 2 groups based on the flow directions
        bbox_flow1, bbox_flow2 = [], []
        for bbox in history_bbox:
            if bbox[7]:
                if bbox[7] == self.flow_direct1:
                    bbox_flow1.append([(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2])
                elif bbox[7] == self.flow_direct2:
                    bbox_flow2.append([(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2])

        return bbox_flow1, bbox_flow2


    def maha_distance(self, bbox_flow1, bbox_flow2, bbox_current_frame):
        """
        compute the mahalanobis distance between each bbox and its flow centre derived from previous 19 frames
        @param bbox_flow1: history bbox within the first flow, nested list [[xc, yc], []...[]]
        @param bbox_flow2: history bbox within the second flow,,nested list [[xc, yc], []...[]]
        @param bbox_current_frame: nested list, [[x1, y1, x2, y2, id, speed, motion_vec, motion_direction], []...[]]
        @return:
            bbox_current_frame: nested list, [[x1, y1, x2, y2, id, speed, motion_vec, motion_direction, in/out], []...[]]
        """

        # mean and covariance
        flow1_array = np.array(bbox_flow1, dtype=float)
        flow2_array = np.array(bbox_flow2, dtype=float)
        mean1, mean2 = np.mean(flow1_array.T, axis=1), np.mean(flow2_array.T, axis=1)
        cov1, cov2 = np.cov(flow1_array.T), np.cov(flow2_array.T)  # cov = L * L.trans

        # TODO: consider the situation when covariance matrix is not positive definite
        L1, L2 = np.linalg.cholesky(cov1), np.linalg.cholesky(cov2)

        for bbox in bbox_current_frame:
            # directly assign 'in' to the current bbox if last bbox with same ID is 'in' the main flow
            for last_bbox in self.bbox_history_ls[-1]:
                if bbox[4] == last_bbox[4]:
                    if last_bbox[8]:
                        if last_bbox[8] == 'in':
                            bbox.append('in')

            # compute Mahala distance between current bbox and history flow centre
            if len(bbox) == 8:
                # if bbox has the same motion direction with flow 1
                if bbox[7] == self.flow_direct1:
                    xc = (bbox[0] + bbox[2]) / 2
                    yc = (bbox[1] + bbox[3]) / 2
                    delta = np.array([xc, yc]) - mean1
                    z = scipy.linalg.solve_triangular(L1, delta.T, lower=True, check_finite=False, overwrite_b=True)
                    squared_maha = np.sum(z * z, axis=0)
                # if bbox has the same motion direction with flow 2
                elif bbox[7] == self.flow_direct2:
                    xc = (bbox[0] + bbox[2]) / 2
                    yc = (bbox[1] + bbox[3]) / 2
                    delta = np.array([xc, yc]) - mean2
                    z = scipy.linalg.solve_triangular(L2, delta.T, lower=True, check_finite=False, overwrite_b=True)
                    squared_maha = np.sum(z * z, axis=0)
                # if bbox has different motion direction with flow 1 and flow 2, mark as 'out'
                else:
                    squared_maha = 100
                    print(bbox)

                if squared_maha <= 5.99*2:    # 95% confidence interval belongs to the chi-squared distribution
                    bbox.append('in')
                else:
                    bbox.append('out')

        return bbox_current_frame