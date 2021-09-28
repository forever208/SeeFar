from traffic_analyst import TrafficAnalyst
import cv2
import argparse


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
    parser.add_argument('--colab', dest='use_colab',
                        help='whether use colab',
                        action='store_true')
    args = parser.parse_args()
    return args


def main():
    TERMINAL = parse_args()
    model = TERMINAL.model
    cap = cv2.VideoCapture(TERMINAL.video_path)
    fps = int(cap.get(5))
    print('fps of loaded video:', fps)
    print('video width:', int(cap.get(3)))
    print('video height:', int(cap.get(4)))

    traffic_analyst = TrafficAnalyst(model, int(cap.get(3)), int(cap.get(4)))
    t = int(1000 / fps)
    videoWriter = None

    while True:
        _, im = cap.read()    # read a frame from video
        if im is None:
            break

        # get result of object tracking, define the size of the video to be saved
        result = traffic_analyst.update(im)
        result = result['frame']    # image that has been added bbox and text, 3D array (h, w, 3)
        # result = imutils.resize(result, height=500)    # resize the result if you want

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
