from traffic_analyst import TrafficAnalyst
import cv2
import argparse
import imutils

def parse_args():
    """
    Parse input arguments from terminal
    """
    parser = argparse.ArgumentParser(description='run a DeepSORT demo')
    parser.add_argument('--video', dest='video_path',
                        help='the local path of the video',
                        default='./video/test_traffic.mp4', type=str)
    parser.add_argument('--output', dest='output_path',
                        help='the output path of the detected video',
                        default='./video/result.mp4', type=str)
    parser.add_argument('--model', dest='model',
                        help='the model name of object detector',
                        default='yolov5m', type=str)
    parser.add_argument('--cam_angle', dest='cam_angle',
                        help='the tile angle of the camera, vary from 0 (top view) to 90 degree (horizontal)',
                        default=0, type=int)
    parser.add_argument('--drone_h', dest='drone_h',
                        help='meters, the height of the flying drone',
                        default=100, type=int)
    parser.add_argument('--drone_speed', dest='drone_speed',
                        help='km/h, the speed of the flying drone',
                        default=0, type=int)
    parser.add_argument('--test_mode', dest='test_mode',
                        help='whether activate the test_mode for speed estimator and flow analysor',
                        action='store_true')
    parser.add_argument('--colab', dest='use_colab',
                        help='whether use colab',
                        action='store_true')
    args = parser.parse_args()
    return args


def main():
    TERMINAL = parse_args()
    model = TERMINAL.model
    cam_angle = TERMINAL.cam_angle
    drone_h = TERMINAL.drone_h
    drone_speed = TERMINAL.drone_speed
    test = TERMINAL.test_mode
    video_num = int(TERMINAL.video_path[8])

    cap = cv2.VideoCapture(TERMINAL.video_path)
    fps = int(cap.get(5))
    video_w = int(cap.get(3))
    video_h = int(cap.get(4))
    print('fps of loaded video:', fps)
    print('video width:', video_w)
    print('video height:', video_h)

    traffic_analyst = TrafficAnalyst(model, video_w, video_h, cam_angle, drone_h, drone_speed, fps, test, video_num)
    t = int(1000 / fps)
    videoWriter = None
    num = 1

    while True:
        _, im = cap.read()    # read a frame from video
        if im is None:
            break

        # num += 1
        # if num % fps == 0:
        #     cv2.imwrite('./video/images2/8_' + str(num) + '.jpg', im)

        # get result of object tracking, define the size of the video to be saved
        result = traffic_analyst.update(im)
        result = result['frame']    # image that has been added bbox and text, 3D array (h, w, 3)
        result = imutils.resize(result, height=1000)    # resize the result if you want

        # save video into RESULT_PATH
        if videoWriter is None:
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')    # opencv3.0
            videoWriter = cv2.VideoWriter(TERMINAL.output_path, fourcc, fps, (result.shape[1], result.shape[0]))
        videoWriter.write(result)

        # do not show image if using Colab because Colab doesn't support cv2.imshow()
        if not TERMINAL.use_colab:
            cv2.imshow('demo', result)
            cv2.waitKey(t)
            if cv2.getWindowProperty('demo', cv2.WND_PROP_AUTOSIZE) < 1:
                break

    cap.release()
    videoWriter.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
