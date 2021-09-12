# vim: expandtab:ts=4:sw=4
import numpy as np
import scipy.linalg


"""
Table for the 0.95 quantile of the chi-square distribution with N degrees of
freedom (contains values for N=1, ..., 9). Taken from MATLAB/Octave's chi2inv
function and used as Mahalanobis gating threshold.
"""
chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919}


class KalmanFilter(object):
    """
    A simple Kalman filter for tracking bounding boxes along image frames.
        1. predict the state for next frame
        2. update predictions by combing YOLO detections

    The 8-dimensional state space (x, y, a, h, vx, vy, va, vh)
    contains the bounding box center position (x, y), aspect ratio a, height h, and their respective velocities.

    Object motion follows a constant velocity model.
    The bbox location (x, y, a, h) is taken as direct observation of the state space (linear observation model).
    """

    def __init__(self):
        ndim, dt = 4, 1.

        # State transfer matrix F, 8*8 (this matrix won't change along the time)
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt

        # Measurement matrix H, 4*8, transform track space into detection space
        self._update_mat = np.eye(ndim, 2 * ndim)

        # Motion and observation uncertainty are chosen relative to the current state estimate.
        # These weights control the amount of uncertainty in the model. This is a bit hacky.
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160


    def initiate(self, measurement):
        """
        Initialise a track for unmatched detection, compute its mean and covariance of the state distribution
        @param measurement: a bbox coordinates (x, y, a, h) with center position (x, y), aspect ratio a, and height h.
        @return:
             mean: 1D array (8, ), mean vector of the new track
             covariance: 2D array (8 ,8), covariance matrix of the new track
        """

        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)    # velocities are initialized to 0 mean
        mean = np.r_[mean_pos, mean_vel]    # concatenate [x, y, a, h] and [vx, vy, va, vh]

        # Initialise covariance matrix P (8x8) by detections
        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3]
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance


    def predict(self, mean, covariance):
        """
        Run Kalman filter prediction based on mean and covariance from last time step

        @param mean: 1D array (8, ),  mean vector of the object state at the previous time step.
        @param covariance: 2D array (8 ,8), covariance matrix of the object state at the previous time step.
        @return:
            mean: 1D array (8, ), predicted mean vector of the next time step
            covariance: 2D array (8 ,8), predicted covariance matrix of the next time step
        """

        # Initialise noise covariance matrix Q (i.e. motion_cov) based on mean and covariance of time t-1
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3]]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3]]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        """ predict state at time t by x' = Fx (1) """
        # x is the state at time t-1 (i.e. mean)
        # F is state transfer matrix (i.e. self._motion_mat)
        mean = np.dot(self._motion_mat, mean)

        """ update uncertainty matrix at time t by P' = FPF^T+Q (2) """
        # P is the covariance at time t-1，Q is the process noise matrix (initialised as a small value)
        covariance = np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        return mean, covariance


    def project(self, mean, covariance):
        """
        Project state space to measurement space.
        @param mean: 1D array (8, ),  mean vector of bbox in state space
        @param covariance: 2D array (8 ,8), covariance matrix of bbox in state space
        @return:
            mean: mean vector of bbox in measurement space
            covariance + innovation_cov: covariance matrix of bbox in measurement spacee
        """

        # Initialise measurement noise matrix R which is a 4*4 diagonal matrix
        # 对角线上的值分别为中心点两个坐标以及宽高的噪声，
        # 以任意值初始化，一般设置宽高的噪声大于中心点的噪声，
        # 该公式先将协方差矩阵P'映射到检测空间，然后再加上噪声矩阵R；
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3]]
        innovation_cov = np.diag(np.square(std))    # noise matrix R

        # transform mean from state space into measurement space by Hx'
        mean = np.dot(self._update_mat, mean)

        # transform covariance from state space into measurement space by S = HP'H^T + R (4)
        covariance = np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T))

        return mean, covariance + innovation_cov


    def update(self, mean, covariance, measurement):
        """
        Run Kalman filter correction step

        @param mean: 1D array (8, ),  mean vector of the object state at current time step
        @param covariance: 2D array (8 ,8), covariance matrix of the object state at current time step
        @param measurement: 1D array (4, ), detection vector vector [x, y, a, h] at current time step
        @return:
            new_mean: 1D array (8, ), updated mean vector of current time step
            new_covariance: 2D array (8 ,8), updated covariance matrix of current time step
        """

        """ transform predictions from state space to measurement space, get Hx' and S (S = HP'H^T + R) (4) """
        projected_mean, projected_cov = self.project(mean, covariance)

        # 矩阵分解
        chol_factor, lower = scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)
        # 计算卡尔曼增益K；相当于求解公式(5)
        # 公式5计算卡尔曼增益K，卡尔曼增益用于估计误差的重要程度
        # 求解卡尔曼滤波增益K 用到了cholesky矩阵分解加快求解；
        # 公式5的右边有一个S的逆，如果S矩阵很大，S^-1求解消耗时间太大，
        # 所以把公式两边同时乘上 S，右边的 S*S=I，转化成 SK=P'H^T 求解 K。
        """ compute Kalman gain by K = P'H^TS^-1 (5) """
        kalman_gain = scipy.linalg.cho_solve((chol_factor, lower), np.dot(covariance, self._update_mat.T).T, check_finite=False).T

        """ compute the difference between prediction and measurement by y = z - Hx' (3) """
        # 在公式3中，z为detection的均值向量，不包含速度变化值，即z=[cx, cy, r, h]，
        # H称为测量矩阵，它将track的均值向量x'映射到检测空间，该公式计算detection和track的均值误差
        innovation = measurement - projected_mean

        """ update state vector by x = x' + Ky (6) """
        new_mean = mean + np.dot(innovation, kalman_gain.T)

        """ update covariance by P = (I - KH)P' (7) """
        new_covariance = covariance - np.linalg.multi_dot((kalman_gain, projected_cov, kalman_gain.T))

        return new_mean, new_covariance


    def gating_distance(self, mean, covariance, measurements,only_position=False):
        """
        Compute Mahala distance between track state and measurements (detections).

        A suitable distance threshold can be obtained from `chi2inv95`.
        the chi-square distribution has 4 DOF if `only_position` is False, otherwise 2.

        @param mean: 1D array (8, ), mean value of the state vector of a track
        @param covariance: 2D array (8,8), covariance matrix of the track state distribution
        @param measurements: 2D array (num_detections, 4), each row is [x, y, a, h] of a bbox detection.
        @param only_position: boolean, if True, distance computation only consider the [x,y] of a bbox
        @return:
            squared_maha: 1D array (num_detections, ), each element is the squared Mahalanobis distance
                          between track (mean, covariance) and `measurements[i]`.
        """

        # Project state space to measurement space.
        mean, covariance = self.project(mean, covariance)

        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        cholesky_factor = np.linalg.cholesky(covariance)
        d = measurements - mean
        z = scipy.linalg.solve_triangular(cholesky_factor, d.T, lower=True, check_finite=False, overwrite_b=True)
        squared_maha = np.sum(z * z, axis=0)

        return squared_maha
