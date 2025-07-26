import cv2
import pathlib
import sys
import time
import utils as util



from ...common import Processor


def main(video_sources):
    # Load model
    weights_path = str(pathlib.Path(__file__).resolve().parents[1] / "weights" / "yolov5s.onnx")
    net = cv2.dnn.readNetFromONNX(weights_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    ln = []

    # Load video captures
    caps = []
    for src in video_sources:
        video_path = str(pathlib.Path(__file__).resolve().parents[2] / src)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Failed to read {src}")
            print("Error code:", cap.get(cv2.CAP_PROP_POS_MSEC))
            return
        else:
            print(f"Successfully read {src}")
        caps.append(cap)

    while True:
        frames = []
        for cap in caps:
            ret, frame = cap.read()
            if not ret:
                print("One of the videos has ended or failed to read.")
                return
            frames.append(frame)

        lanes = util.Lanes([
            util.Lane(0, frames[0], 1),
            util.Lane(0, frames[1], 2),
            util.Lane(0, frames[2], 3),
            util.Lane(0, frames[3], 4)
        ])

        # Run detection and update lanes
        lanes = util.final_output_onnx(net, ln, lanes)
        wait_time = util.schedule(lanes)
        output = util.display_result(wait_time, lanes)

        cv2.imshow("Smart Traffic Control System", output)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release all videos
    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    sources = [
        "video0_resized.mp4",
        "video1_resized.mp4",
        "video2_resized.mp4",
        "video3_resized.mp4",
        "video4_resized.mp4"
    ]
    print("Using video sources:", sources)
    main(sources)
