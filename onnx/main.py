import cv2
import time
import numpy as np
import argparse
import pathlib
import os

import utils as util






def test_video(vs, video_name):
    """
    Test function to check if video is successfully read.
    """
    success, frame = vs.read()
    if not success:
        print(f"Failed to read {video_name}")
        print(f"Error code: {vs.get(cv2.CAP_PROP_POS_FRAMES)}")  # Get error code
    else:
        print(f"Successfully read {video_name}")

def main(sources):
    # Load each lane's video source from the 'datas/' folder
    data_path = pathlib.Path(__file__).resolve().parents[1] / "datas"

    vs1 = cv2.VideoCapture(str(data_path / sources[0]))
    vs2 = cv2.VideoCapture(str(data_path / sources[1]))
    vs3 = cv2.VideoCapture(str(data_path / sources[2]))
    vs4 = cv2.VideoCapture(str(data_path / sources[3]))
    vs5 = cv2.VideoCapture(str(data_path / sources[4]))

    # Test the video loading for each video file
    test_video(vs1, sources[0])
    test_video(vs2, sources[1])
    test_video(vs3, sources[2])
    test_video(vs4, sources[3])
    test_video(vs5, sources[4])

    # Load YOLO ONNX model
    net = cv2.dnn.readNet(str(pathlib.Path(__file__).resolve().parents[1] / "weights" / "yolov5s.onnx"))
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    ln = net.getUnconnectedOutLayersNames()

    # Initialize lanes
    lanes = util.Lanes([
        util.Lane("", "", 1),
        util.Lane("", "", 2),
        util.Lane("", "", 3),
        util.Lane("", "", 4),
        util.Lane("", "", 5),
    ])
    
    wait_time = 0

    while True:
        success1, frame1 = vs1.read()
        success2, frame2 = vs2.read()
        success3, frame3 = vs3.read()
        success4, frame4 = vs4.read()
        success5, frame5 = vs5.read()

        if not (success1 and success2 and success3 and success4 and success5):
            print("One of the videos has ended or failed to read.")
            break

        for lane in lanes.getLanes():
            if lane.lane_number == 1:
                lane.frame = frame1
            elif lane.lane_number == 2:
                lane.frame = frame2
            elif lane.lane_number == 3:
                lane.frame = frame3
            elif lane.lane_number == 4:
                lane.frame = frame4
            elif lane.lane_number == 5:
                lane.frame = frame5

        start = time.time()
        lanes = util.final_output(net, ln, lanes)
        end = time.time()
        print("Processing time:", round(end - start, 2), "seconds")

        if wait_time <= 0:
            transition_img = util.display_result(wait_time, lanes)
            final_img = cv2.resize(transition_img, (1020, 720))
            cv2.imshow("Traffic Flow", final_img)
            cv2.waitKey(100)

            wait_time = util.schedule(lanes)

        display_img = util.display_result(wait_time, lanes)
        final_img = cv2.resize(display_img, (1020, 720))
        cv2.imshow("Traffic Flow", final_img)
        cv2.waitKey(1)
        wait_time -= 1

    vs1.release()
    vs2.release()
    vs3.release()
    vs4.release()
    vs5.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Traffic Control System")
    parser.add_argument(
        "--sources",
        help="Comma-separated video files located in datas/ folder",
        type=str,
        default="video0_resized.mp4,video1_resized.mp4,video2_resized.mp4,video3_resized.mp4,video4_resized.mp4"
    )
    args = parser.parse_args()
    sources = args.sources.split(",")
    print("Using video sources:", sources)
    main(sources)
