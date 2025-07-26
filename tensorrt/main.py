import sys
import pathlib
import cv2
import time
import numpy as np
from Processor import Processor
import argparse

# Add the common directory to the Python path
sys.path.insert(1, str(pathlib.Path.cwd().parents[0]) + "/common")
from common import utils as util

def main(sources):
    # Read image from each lane's video source
    vs = cv2.VideoCapture(str(pathlib.Path.cwd().parents[0]) + "/datas/" + sources[0])
    vs2 = cv2.VideoCapture(str(pathlib.Path.cwd().parents[0]) + "/datas/" + sources[1])
    vs3 = cv2.VideoCapture(str(pathlib.Path.cwd().parents[0]) + "/datas/" + sources[2])
    vs4 = cv2.VideoCapture(str(pathlib.Path.cwd().parents[0]) + "/datas/" + sources[3])
    print("Working directory:", str(pathlib.Path.cwd().parents[0]))

    # Generate a TensorRT engine with the given model
    processor = Processor("yolov5s.trt")

    # Initial configuration of each lane's order
    lanes = util.Lanes([util.Lane("", "", 1), util.Lane("", "", 2), util.Lane("", "", 3), util.Lane("", "", 4)])
    wait_time = 0

    while True:
        # Read the next frame from each video source
        (success, frame) = vs.read()
        (success, frame2) = vs2.read()
        (success, frame3) = vs3.read()
        (success, frame4) = vs4.read()

        # If the frame was not successfully captured, break the loop
        if not success:
            break

        # Assign each lane its corresponding frame
        for lane in lanes.getLanes():
            if lane.lane_number == 1:
                lane.frame = frame
            elif lane.lane_number == 2:
                lane.frame = frame2
            elif lane.lane_number == 3:
                lane.frame = frame3
            elif lane.lane_number == 4:
                lane.frame = frame4

        # Process the frames with TensorRT
        start = time.time()
        lanes = util.final_output_tensorrt(processor, lanes)  # Returns lanes object with processed frames
        end = time.time()
        print("Total processing time:", str(end - start))

        # Display the results
        if wait_time <= 0:
            images_transition = util.display_result(wait_time, lanes)    
            final_image = cv2.resize(images_transition, (1080, 720))
            cv2.imshow("Traffic Control System", final_image)
            cv2.waitKey(10)

            # Get the waiting duration of each lane
            wait_time = util.schedule(lanes)

        # Display the scheduled result
        images_scheduled = util.display_result(wait_time, lanes)    
        final_image = cv2.resize(images_scheduled, (1080, 720))
        cv2.imshow("Traffic Control System", final_image)
        cv2.waitKey(1)

        # Decrease the wait time for the next iteration
        wait_time -= 1

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Determines duration based on car count in videos")
    parser.add_argument("--sources", default="video1.mp4,video5.mp4,video2.mp4,video3.mp4", help="Comma-separated list of video feeds to be processed. The videos must reside in the 'datas' folder.")
    args = parser.parse_args()

    # Split the video sources argument into a list
    sources = args.sources.split(",")
    print("Video sources:", sources)

    # Call the main function with the video sources
    main(sources)
