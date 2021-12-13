"""
old version of flow analysor, has been deprecated
"""

import numpy as np
import math


class DynamicArea():
    """
    The dynamic area is designed by 4 corners point (i.e top_left, top_right, bottom_left, bottom_right)
    """

    def __init__(self, num_frames=40):
        self.bbox_history_ls = []
        self.num_frames = num_frames    # number of frames the dynamic area maintains
        self.init_dynamic_area = False
        self.top_left = None
        self.top_right = None
        self.bottom_left = None
        self.bottom_right = None
        self.center = None
        self.diameter = None


    def update_area(self, current_frame_bbox):
        """
        compute the dynamic area after each new frame comes in
        @param current_frame_bbox: nested list, [[x1, y1, x2, y2, id, speed, motion_vec, motion_direction], []...[]]
        @return:
            bbox_within_area: nested list, [[x1, y1, x2, y2, id, speed, motion_vec, motion_direction, in/out], []...[]]
        """

        bbox_current_array = self.list_to_2Darray(current_frame_bbox)    # [[xc, yc], []...[]]

        if bbox_current_array.size >= 5:
            # add bbox of current frame into the bbox history
            self.bbox_history_ls.append(bbox_current_array)

            # do dynamic management if reaching full history frames
            if len(self.bbox_history_ls) == self.num_frames:

                # transform total bbox list into 2D array
                bbox_history_array = self.bbox_history_ls[0]
                for i in range(1, self.num_frames-1):
                    bbox_history_array = np.concatenate((bbox_history_array, self.bbox_history_ls[i]), axis=0)


                """Area computation for old bbox history """
                # find the 4 corners coordinates from New history
                old_l_top, old_r_top, old_l_bottom, old_r_bottom = self.find_four_corners(bbox_history_array)

                # initialise the dynamic area
                if not self.init_dynamic_area:
                    self.top_left = old_l_top
                    self.top_right = old_r_top
                    self.bottom_left = old_l_bottom
                    self.bottom_right = old_r_bottom
                    self.center, self.diameter = self.find_center(self.top_left, self.top_right, self.bottom_left,
                                                                  self.bottom_right)
                    self.init_dynamic_area = True

                """remove outliers for current frame, update the dynamic area"""
                self.bbox_history_ls[self.num_frames-1] = self.remove_outlier(bbox_current_array)

                # add current frame bbox into bbox history
                bbox_history_array = np.concatenate((bbox_history_array, self.bbox_history_ls[self.num_frames-1]), axis=0)

                # update the dynamic area
                new_l_top, new_r_top, new_l_bottom, new_r_bottom = self.find_four_corners(bbox_history_array)
                self.top_left = new_l_top
                self.top_right = new_r_top
                self.bottom_left = new_l_bottom
                self.bottom_right = new_r_bottom
                self.center, self.diameter = self.find_center(self.top_left, self.top_right, self.bottom_left,
                                                              self.bottom_right)
                del self.bbox_history_ls[0]

        bbox_within_area = current_frame_bbox

        return bbox_within_area


    def find_four_corners(self, bbox_2D_array):
        """
        find the 4 corners
        @param bbox_2D_array: 2D array, all bbox of the latest 20 frames [[xc, yc], [], []]
        @return:
            l_top: 1D array, [x, y]
            r_top: 1D array, [x, y]
            l_bottom: 1D array, [x, y]
            r_bottom: 1D array, [x, y]
        """

        x_plus_2y = bbox_2D_array[:, 0] + 2*bbox_2D_array[:, 1]    # 1D array
        x_plus_y = bbox_2D_array[:, 0] + bbox_2D_array[:, 1]  # 1D array
        x_subtract_y = bbox_2D_array[:, 0] - bbox_2D_array[:, 1]    # 1D array
        x_subtract_2y = bbox_2D_array[:, 0] - 2*bbox_2D_array[:, 1]  # 1D array

        l_top = bbox_2D_array[np.argmin(x_plus_2y)]    # min(x+2y) is considered as the left_top point
        r_bottom = bbox_2D_array[np.argmax(x_plus_y)]    # max(x+y) is considered as the right_bottom point
        r_top = bbox_2D_array[np.argmax(x_subtract_y)]    # max(x-y) is considered as the right_top point
        l_bottom = bbox_2D_array[np.argmin(x_subtract_2y)]    # min(x-y) is considered as the left_bottom point

        return l_top, r_top, l_bottom, r_bottom


    def find_center(self, l_top, r_top, l_bottom, r_bottom):
        """
        Given 4 corner points, find the minimal circle enclose them
        @param l_top: 1D array, [x, y]
        @param r_top: 1D array, [x, y]
        @param l_bottom: 1D array, [x, y]
        @param r_bottom: 1D array, [x, y]
        @return:
            center: 1D array, [x, y]
            diameter: a float number
        """

        if ((r_bottom[0]-l_top[0])**2 + (r_bottom[1]-l_top[1])**2) > ((l_bottom[0]-r_top[0])**2 + (l_bottom[1]-r_top[1])**2):
            center = [(r_bottom[0] + l_top[0]) / 2, (r_bottom[1] + l_top[1]) / 2]
            diameter = math.sqrt((r_bottom[0]-l_top[0])**2 + (r_bottom[1]-l_top[1])**2)
        else:
            center = [(l_bottom[0] + r_top[0]) / 2, (l_bottom[1] + r_top[1]) / 2]
            diameter = math.sqrt((l_bottom[0]-r_top[0])**2 + (l_bottom[1]-r_top[1])**2)

        return center, diameter


    def is_outlier(self, bbox):
        """
        judge if the new area is changing smoothly from the old area
        @param bbox: 1D array, [x, y]
        """

        # the distance between bbox and old center of dynamic area
        distance_to_center = math.sqrt((bbox[0]-self.center[0])**2 + (bbox[1]-self.center[1])**2)
        distance_ratio  = distance_to_center / (self.diameter/2)

        if distance_ratio < 1.4:
            return False
        else:
            return True


    def remove_outlier(self, bbox_current_array):
        """
        remove the outliers for current frame bbox
        @param bbox_current_array: 2D array, [[x, y], []...[]]
        @return: current frame bbox without outlier, 2D array, [[x, y], []...[]]
        """

        processed_bbox = []
        for each_bbox in bbox_current_array:
            if not self.is_outlier(each_bbox):
                processed_bbox.append(each_bbox)

        return np.array(processed_bbox)


    def list_to_2Darray(self, nested_list):
        """
        transform the bbox format into 2D numpy array
        @param nested_list: nested list, [[x1, y1, x2, y2, '', track_id], []...[]]
        @return:
            2D array (num_bboxes, 2), [[xc, yc], []...[]]
        """

        bbox_list = []
        for bbox in nested_list:
            xc = (bbox[0] + bbox[2]) / 2
            yc = (bbox[1] + bbox[3]) / 2
            bbox_list.append([xc, yc])

        return np.array(bbox_list)
