from ultralytics import YOLO
import numpy as np
import time
import pyrealsense2 as rs
import argparse
import cv2

parser = argparse.ArgumentParser(description='Model real-time params')
parser.add_argument('model', type=int, help='Select the model for inference')
parser.add_argument('conf', type=int, help='Confidence threshold in percentage (2 digits integer)')
parser.add_argument('iou', type=int, help='IoU threshold in percentage (2 digits integer)')
args = parser.parse_args()


rgb_int = {
    'fx': 617.323486328125,
    'fy': 617.6768798828125,
    'ppx': 330.5740051269531,
    'ppy': 235.93508911132812,
}

model_list = ["./models/yolov8n_int8_openvino_model",
              "./models/yolov5n_int8_openvino_model",
              "./models/yolov11n_int8_openvino_model"]


model_path = model_list[args.model]
COI = [0, 56, 63, 64, 66, 67]

def transform(cx, cy, d, rgbint):
    fx = rgbint['fx']
    fy = rgbint['fy']
    ppx = rgbint['ppx']
    ppy = rgbint['ppy']
    
    dcm = 100*d.get_distance(cx, cy)
    cxcm = (cx - ppx) * dcm / fx
    cycm = (cy - ppy) * dcm / fy
    
    return cxcm, cycm, dcm


# Load the OpenVINO model
ov_model = YOLO(model_path, task='detect')

# Configure the pipeline
pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)  # RGB stream
cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # Depth stream

# Start the pipeline
profile = pipe.start(cfg)

# Create an align object
align = rs.align(rs.stream.color)  # Align depth to color

# User-specified confidence threshold
confidence_threshold = args.conf/100
# IoU threshold for Non-Max Suppression
iou_threshold = args.iou/100

while True:
    start_time = time.time()

    # Wait for frames
    frameset = pipe.wait_for_frames()
    # Align frames
    aligned_frames = align.process(frameset)
    
    # Get aligned frames
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    
    if not depth_frame or not color_frame:
        print("Error: Could not read frames.")
        break

    # Convert color frame to numpy array
    frame_rgb = np.asanyarray(color_frame.get_data())
    #frame_depth = np.asanyarray(depth_frame.get_data())

    # Run inference on the current frame
    results = ov_model(frame_rgb)

    # Initialize lists for storing bounding boxes, confidences, and class IDs
    boxes = []
    confidences = []
    class_ids = []
    frame_report = []

    # Process results and collect boxes, confidences, and class IDs
    for result in results:
        result_boxes = result.boxes.xyxy  # Get bounding box coordinates
        result_confidences = result.boxes.conf  # Get confidence scores
        result_class_ids = result.boxes.cls  # Get class IDs

        for box, conf, class_id in zip(result_boxes, result_confidences, result_class_ids):
            if conf >= confidence_threshold and class_id in COI:  # Filter out low-confidence detections
                x1, y1, x2, y2 = map(int, box)  # Convert to int
                boxes.append([x1, y1, x2 - x1, y2 - y1])  # Convert to format for cv2.dnn.NMSBoxes
                confidences.append(float(conf))  # Convert confidence to float
                class_ids.append(int(class_id))  # Convert class ID to int

    # Apply Non-Max Suppression (NMS)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, iou_threshold)

    # To create the report for each frame
    if len(indices) > 0:
        for i in indices.flatten():
            x1, y1, w, h = boxes[i]

            # Get the center of the bounding box
            center_x = x1 + int(w / 2)
            center_y = y1 + int(h / 2)

            center_x_cm, center_y_cm, depth_cm = transform(center_x, center_y, depth_frame, rgb_int)

            # Append object data to the report
            frame_report.append([center_x_cm, center_y_cm, depth_cm, class_ids[i], f"{confidences[i]:.2f}"])

    print(frame_report)

    # Calculate average inference time
    ex_time = time.time() - start_time
    print(f"Absolute inference time: {ex_time:.4f} seconds")

# Release the pipeline
pipe.stop()
print("Done!")
