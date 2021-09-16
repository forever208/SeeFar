import numpy as np
import math


class SpeedEstimation:
    """
    The vehicle speed is estimated by the geometry of camera imaging
    """

    def __init__(self):
        self.bbox_history = []
        self.speed_info_his = {}    # speed info of 100 history tracks, {track_id: (motion_vec, speed)...}
        self.num_frames = 10    # number of frames when the car displacement is calculated
        self.param_init = False
        self.img_w = None
        self.img_h = None
        self.cam_FOV = (94 * math.pi) / 180    # 94 degree, the FOV of the camera
        self.cam_angle = (20 * math.pi) / 180    # 10 degree, the angle of the camera regarding to down view angle
        self.cam_focal_len = 3.61    # mm, the focal length of the camera
        self.cam_sensor_w = 6.17    # mm, the actual size of the camera sensor
        self.cam_sw = None    # camera_sensor_with / pixel_width, means the size of a pixel in real world
        self.drone_H = 100    # meter, the real height of the drone (camera)
        self.fps = 30    # frame rate of the video


    def speed_update(self, current_frame_bbox, image):
        """
        compute the dynamic area after each new frame comes in
        @param current_frame_bbox: nested list, [[x1, y1, x2, y2, track_id], []...[]]
        @param image: original video frame, 3D array (h, w, 3)
        @return:
            bbox_with_speed: nested list, [[x1, y1, x2, y2, id, speed, motion_vec, motion_direction], []...[]]
        """

        if not self.param_init:
            self.img_h = image.shape[0]
            self.img_w = image.shape[1]
            self.cam_sw = self.cam_sensor_w / self.img_w    # mm/pixel
            self.param_init = True

        bbox_with_speed = current_frame_bbox
        if len(current_frame_bbox) > 0:
            self.bbox_history.append(current_frame_bbox)

            if len(self.bbox_history) == self.num_frames:
                for tnow_bbox in self.bbox_history[self.num_frames - 1]:
                    bbox_matched = False

                    # for each bbox in current frame, find its corresponding bbox in previous 10th frames
                    for t0_bbox in self.bbox_history[0]:
                        if t0_bbox[4] == tnow_bbox[4]:
                            bbox_matched = True
                            speed, motion_vec = self.speed_calculation(t0_bbox, tnow_bbox, self.num_frames)

                            # add track_id: (motion_vec, speed) to dict, maintain a history dict with 100 length
                            self.speed_info_his[tnow_bbox[4]] = (motion_vec, speed)
                            if len(self.speed_info_his) == 100:
                                for key in self.speed_info_his.keys():
                                    del self.speed_info_his[key]
                                    break
                            break

                    # if current track ID has record in history dict, assign history speed info to current track ID
                    if not bbox_matched and self.speed_info_his:
                        for track_id, value in self.speed_info_his.items():
                            if track_id == tnow_bbox[4]:
                                speed = value[1]
                                motion_vec = value[0]
                                bbox_matched = True

                    if not bbox_matched:
                        speed = ''
                        motion_vec = ''

                    tnow_bbox.append(speed)
                    tnow_bbox.append(motion_vec)

                bbox_with_speed = self.bbox_history[-1]
                del self.bbox_history[0]

            # fill in with None for speed and motion_vec when frames have not been accumulated to full
            else:
                for bbox in current_frame_bbox:
                    bbox.append('')
                    bbox.append('')
                bbox_with_speed = current_frame_bbox

            # add direction: [x1, y1, x2, y2, id, speed, motion_vec] --> [x1, y1, x2, y2, id, speed, motion_vec, 'up']
            bbox_with_speed = self.motion_direction(bbox_with_speed)

        return bbox_with_speed


    def speed_calculation(self, t0_bbox, tnow_bbox, t_interval):
        """
        compute the speed of each bbox within the duration of t_interval
        @param t0_bbox: bbox of 5 frames ahead, nested list, [[x1, y1, x2, y2, track_id, speed], []...[]]
        @param tnow_bbox: bbox of current frame, nested list, [[x1, y1, x2, y2, track_id, speed], []...[]]
        @param t_interval: int, time interval between t0 and tnow
        @return:
            speed: float, km/h
            motion_vec: float tuple, (x_move, y_move, t_interval)
        """

        # compute the displacement in pixel plane
        x_old = (t0_bbox[0] + t0_bbox[2]) / 2
        y_old = (t0_bbox[1] + t0_bbox[3]) / 2
        x_new = (tnow_bbox[0] + tnow_bbox[2]) / 2
        y_new = (tnow_bbox[1] + tnow_bbox[3]) / 2
        pixel_dis = math.sqrt((x_new - x_old) ** 2 + (y_new - y_old) ** 2)
        motion_vec = (x_new-x_old, y_new-y_old, t_interval)

        # compute the angle from image plane
        w_to_center = abs((x_new + x_old) / 2 - (self.img_w / 2)) * self.cam_sw
        h_to_center = ((y_new + y_old) / 2 - (self.img_h / 2)) * self.cam_sw
        phi = math.atan(h_to_center / self.cam_focal_len)  # can be positive or negative
        h1 = math.sqrt(self.cam_focal_len ** 2 + h_to_center ** 2)
        beta = math.atan(w_to_center / h1)  # can only be positive

        # compute the distance between the drone and tracked car in real world
        H2 = self.drone_H / math.cos(self.cam_angle + phi)  # meter
        D = H2 / math.cos(beta)  # meter
        GSD = ((D * 1000) / self.cam_focal_len) * self.cam_sw  # mm/pixel

        # compute the speed
        speed = (pixel_dis * GSD * 0.001) / ((1.0 / self.fps) * (t_interval - 1))  # m/s
        speed = round(speed * 0.001 * 3600, 1)    # m/s --> km/h

        return speed, motion_vec


    def motion_direction(self, current_frame_bbox):
        """
        add motion direction for each bbox
        @param current_frame_bbox: nested list, [[x1, y1, x2, y2, track_id, speed, motion_vec], []...[]]
        @return:
            current_frame_bbox: nested list, [[x1, y1, x2, y2, track_id, speed, motion_vec, motion_direction], []...[]]
        """
        for i, (x1, y1, x2, y2, id, speed, motion_vec) in enumerate(current_frame_bbox):
            if motion_vec:
                # valid motion: magnitude of motion vector has to be >= 0.25 pixel/frame
                if (abs(motion_vec[0])/motion_vec[2] + abs(motion_vec[1])/motion_vec[2]) >= 0.1:
                    # x displacement > y displacement
                    if abs(motion_vec[0]) > abs(motion_vec[1]):
                        if motion_vec[0] > 0:
                            current_frame_bbox[i].append('right')
                        else:
                            current_frame_bbox[i].append('left')
                    # x displacement < y displacement
                    else:
                        if motion_vec[1] > 0:
                            current_frame_bbox[i].append('down')
                        else:
                            current_frame_bbox[i].append('up')
                else:
                    current_frame_bbox[i].append('static')
            else:
                current_frame_bbox[i].append('')

        return current_frame_bbox

