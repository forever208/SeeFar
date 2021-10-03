import numpy as np
import math


class SpeedEstimation:
    """
    The vehicle speed is estimated by the geometry of camera imaging
    """

    def __init__(self, cam_angle, drone_h, drone_pos, drone_speed, fps):
        self.bbox_history = []
        self.speed_info_his = {}    # speed info of 100 history tracks, {track_id: (motion_vec, speed)...}
        self.num_frames = 10    # number of frames when the car displacement is calculated
        self.param_init = False
        self.img_w = None
        self.img_h = None
        self.cam_FOV = (94 * math.pi) / 180    # 94 degree, the FOV of the camera
        self.cam_angle = (cam_angle * math.pi) / 180  # 10 degree, the angle of the camera regarding to down view angle
        self.cam_focal_len = 3.61    # mm, the focal length of the camera
        self.cam_sensor_w = 6.17    # mm, the actual size of the camera sensor
        self.cam_w_pixel = None    # mm/pixel, camera_sensor_width / pixel_width, means the size of a pixel in real world
        self.drone_h = drone_h    # meter, the real height of the drone (camera)
        self.drone_pos = drone_pos
        self.drone_speed = drone_speed    # km/h, the speed of the drone
        self.fps = fps    # frame rate of the video


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
            self.cam_w_pixel = self.cam_sensor_w / self.img_w    # mm/pixel
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
        @param t0_bbox: bbox of 10 frames ahead, nested list, [[x1, y1, x2, y2, track_id, speed], []...[]]
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
        delta_x = x_new - x_old
        delta_y = y_new - y_old
        pixel_dis = math.sqrt(delta_x ** 2 + delta_y ** 2)
        motion_vec = (delta_x, delta_y)

        # compute the angle from image plane
        w_to_center = abs((x_new + x_old) / 2 - (self.img_w / 2)) * self.cam_w_pixel  # mm
        h_to_center = ((self.img_h - (y_new+y_old)/2) - (self.img_h/2)) * self.cam_w_pixel  # mm
        phi = math.atan(h_to_center / self.cam_focal_len)    # can be positive or negative

        h2 = math.sqrt(self.cam_focal_len ** 2 + h_to_center ** 2)
        beta = math.atan(w_to_center / h2)    # can only be positive
        d = math.sqrt(h2 ** 2 + w_to_center ** 2)

        # compute the distance between the drone and tracked car in real world
        H2 = self.drone_h / math.cos(self.cam_angle + phi)    # meter
        D = H2 / math.cos(beta)    # meter
        GSD = ((D * 1000) / d) * self.cam_w_pixel    # mm/pixel

        # compute the speed of the car
        speed = (pixel_dis * GSD * 0.001) / ((t_interval-1) / self.fps)  # m/s
        speed = round(speed * 0.001 * 3600, 1)    # m/s --> km/h

        # add direction for car speed, compensate the speed by considering the speed of the drone
        if abs(delta_x) > abs(delta_y):
            if delta_x > 0:
                speed = speed + self.drone_speed    # car moves to right, drone moves to right
            else:
                speed = -speed + self.drone_speed    # car moves to left, drone moves to right
        else:
            if delta_y > 0:
                speed = speed + self.drone_speed
            else:
                speed = -speed + self.drone_speed

        return int(speed), motion_vec


    def motion_direction(self, current_frame_bbox):
        """
        add motion direction for each bbox
        @param current_frame_bbox: nested list, [[x1, y1, x2, y2, track_id, speed, motion_vec], []...[]]
        @return:
            current_frame_bbox: nested list, [[x1, y1, x2, y2, track_id, speed, motion_vec, motion_direction], []...[]]
        """
        for i, (x1, y1, x2, y2, id, speed, motion_vec) in enumerate(current_frame_bbox):
            if motion_vec:
                # valid motion: magnitude of motion vector has to be >= 0.1 pixel/frame
                if (abs(motion_vec[0]) + abs(motion_vec[1])) >= 0.1*self.num_frames:
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

