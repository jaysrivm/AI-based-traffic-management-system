import onnxruntime
import numpy as np
import cv2

# Load the ONNX model
session = onnxruntime.InferenceSession("yolov5s.onnx")

# Load and preprocess an image
image = cv2.imread("sample.jpg")  # Replace with your image
image_resized = cv2.resize(image, (640, 640))
image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
input_image = image_rgb.transpose(2, 0, 1) / 255.0  # Channels-first and normalize
input_tensor = input_image[np.newaxis, :].astype(np.float32)

# Run inference
input_name = session.get_inputs()[0].name
outputs = session.run(None, {input_name: input_tensor})

# Print raw outputs (just to see it works)
print("Model output:")
print(outputs[0])
