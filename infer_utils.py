"""
Created on Fri Nov 15 15:51:11 2024

@author: arash
"""
import openvino as ov
import openvino.properties.intel_cpu as intel_cpu
import openvino.properties.hint as hints
import pyrealsense2 as rs
import cv2
import numpy as np
from pathlib import Path
import re
import yaml


def yaml_load(file="data.yaml", append_filename=False):
    assert Path(file).suffix in {".yaml", ".yml"}, f"Attempting to load non-YAML file {file} with yaml_load()"
    with open(file, errors="ignore", encoding="utf-8") as f:
        s = f.read()  # string
        # Remove special characters
        if not s.isprintable():
            s = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFFFD\U00010000-\U0010ffff]+", "", s)
        # Add YAML filename to dict and return
        data = yaml.safe_load(s) or {}  # always return a dict (yaml.safe_load() may return None for empty files)
        if append_filename:
            data["yaml_file"] = str(file)
        return data



class FlexInfer():    
    def __init__(self, ):
        self.DEVICE = "CPU"
        self.yaml_path = "./coco.yaml"
        self.CLASSES = yaml_load(self.yaml_path)["names"]
        self.colors = np.random.uniform(0, 255, size=(len(self.CLASSES), 3))
        self.s_nms = 0.45
        self.rgb_ints = {
            'fx': 617.323486328125,
            'fy': 617.6768798828125,
            'ppx': 330.5740051269531,
            'ppy': 235.93508911132812,
            }   
        
    def RS(self):
        """
        This function initialize the stream of realsense camera
        """

        pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)  # RGB stream
        cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # Depth stream
            
        # Start the pipeline
        profile = pipe.start(cfg)
        device = profile.get_device()
        depth_sensor = device.first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        
        
        # Create an align object
        align = rs.align(rs.stream.color)  # Align depth to color
            
        return pipe, align, depth_scale
    
            
    def WC(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            
        return cap
    
    
    
    
    def cmodel(self, model):
        
        model_path = f"./models/{model}_openvino_model/{model}.xml"
        core = ov.Core()
        core.set_property("CPU", intel_cpu.denormals_optimization(True))
        core.set_property("CPU", {hints.execution_mode: hints.ExecutionMode.PERFORMANCE})
        
        # Uncomment if accuracy matters
        #core.set_property("CPU", {hints.execution_mode: hints.ExecutionMode.ACCURACY})
        
        compiled_model = core.compile_model(model_path, device_name=self.DEVICE, config=self.CONFIG)
        #compiled_model = core.compile_model(model_path, device_name=device)
        
        
        return compiled_model

    def wc_preprocess(self, frame, input_tuple):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (input_tuple[2], input_tuple[3]))
        frame = frame.astype(np.float32) / 255.0
        frame = np.transpose(frame, (2, 0, 1))  # (3, 640, 640)
        frame = np.expand_dims(frame, axis=0)
        
        return frame
    
    
    def rs_preprocess(self, color, depth, input_tuple, depth_scale):
        color = np.asanyarray(color.get_data())
        depth = np.asanyarray(depth.get_data())
        depth = depth * depth_scale
        color = cv2.resize(color, (input_tuple[2], input_tuple[3]))
        depth = cv2.resize(depth, (input_tuple[2], input_tuple[3]))
        color = color.astype(np.float32) / 255.0
        color = np.transpose(color, (2, 0, 1))  # (3, 640, 640)
        color = np.expand_dims(color, axis=0)
        
        return color, depth
    
    
    def intrinsic(self, box_center, depth, rgb_ints):
        fx = rgb_ints['fx']
        fy = rgb_ints['fy']
        ppx = rgb_ints['ppx']
        ppy = rgb_ints['ppy']
        
        dcm = 100*depth[box_center[0], box_center[1]]
        cxcm = (box_center[0] - ppx) * dcm / fx
        cycm = (box_center[1] - ppy) * dcm / fy
        
        return cxcm, cycm, dcm
    
    
    def draw_bounding_box(self, img, class_id, confidence, x, y, x_plus_w, y_plus_h):
        """
        Draws bounding boxes on the input image based on the provided arguments.

        Args:
            img (numpy.ndarray): The input image to draw the bounding box on.
            class_id (int): Class ID of the detected object.
            confidence (float): Confidence score of the detected object.
            x (int): X-coordinate of the top-left corner of the bounding box.
            y (int): Y-coordinate of the top-left corner of the bounding box.
            x_plus_w (int): X-coordinate of the bottom-right corner of the bounding box.
            y_plus_h (int): Y-coordinate of the bottom-right corner of the bounding box.
        """
        
        label = f"{self.CLASSES[class_id]} ({confidence:.2f})"
        color = self.colors[class_id]
        cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
        cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    
    
    def od_vis(self, original_image, res, conf, iou):
        outputs = np.array([cv2.transpose(res[0])])
        rows = outputs.shape[1]

        boxes = []
        scores = []
        class_ids = []
        
        original_image = np.array(255*original_image, dtype=np.uint8)
        original_image = np.transpose(original_image[0], (1, 2, 0))
        
            # Iterate through output to collect bounding boxes, confidence scores, and class IDs
        for i in range(rows):
            classes_scores = outputs[0][i][4:]
            (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
            if maxScore >= conf:
                box = [
                    outputs[0][i][0] - (0.5 * outputs[0][i][2]),
                    outputs[0][i][1] - (0.5 * outputs[0][i][3]),
                    outputs[0][i][2],
                    outputs[0][i][3],
                ]
                boxes.append(box)
                scores.append(maxScore)
                class_ids.append(maxClassIndex)

            # Apply NMS (Non-maximum suppression)
        result_boxes = cv2.dnn.NMSBoxes(boxes, scores, conf, iou, self.s_nms)

        detections = []

            # Iterate through NMS results to draw bounding boxes and labels
        for i in range(len(result_boxes)):
            index = result_boxes[i]
            box = boxes[index]
            detection = {
                "class_id": class_ids[index],
                "class_name": self.CLASSES[class_ids[index]],
                "confidence": scores[index],
                "box": box,
            }
            detections.append(detection)
            
            self.draw_bounding_box(
                original_image,
                class_ids[index],
                scores[index],
                round(box[0]),
                round(box[1]),
                round((box[0] + box[2])),
                round((box[1] + box[3])),
            )

        return original_image


    def od_report(self, res, conf, iou):
        outputs = np.array([cv2.transpose(res[0])])
        rows = outputs.shape[1]

        boxes = []
        scores = []
        class_ids = []
        
            # Iterate through output to collect bounding boxes, confidence scores, and class IDs
        for i in range(rows):
            classes_scores = outputs[0][i][4:]
            (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
            if maxScore >= conf:
                box = [
                    outputs[0][i][0] - (0.5 * outputs[0][i][2]),
                    outputs[0][i][1] - (0.5 * outputs[0][i][3]),
                    outputs[0][i][2],
                    outputs[0][i][3],
                ]
                boxes.append(box)
                scores.append(maxScore)
                class_ids.append(maxClassIndex)

            # Apply NMS (Non-maximum suppression)
        result_boxes = cv2.dnn.NMSBoxes(boxes, scores, conf, iou, self.s_nms)

        detections = []
        for i in range(len(result_boxes)):
            box = boxes[i]
            box_center_x = round(box[0]+box[2]/2)
            box_center_y = round(box[1]+box[3]/2)
            index = result_boxes[i]
            name = self.CLASSES[class_ids[index]]
            confidence = scores[index]
            detections.append([box_center_x, box_center_y, name, f"{confidence:.2f}"])
            
        detection_strings = []

        for detection in detections:
            name = detection[2]
            x = detection[0]
            y = detection[1]
            c = detection[3]
            detection_strings.append(f"{name} in ({x}, {y}) with conf: {c}")

        print(", ".join(detection_strings) + "----------------------------------------------")
        
        return detections
        
        
        
        

    def pose_est(self, res, depth, conf, iou):
        outputs = np.array([cv2.transpose(res[0])])
        rows = outputs.shape[1]

        boxes = []
        scores = []
        class_ids = []
        for i in range(rows):
            classes_scores = outputs[0][i][4:]
            (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
            if maxScore >= conf:
                box = [
                    outputs[0][i][0] - (0.5 * outputs[0][i][2]),
                    outputs[0][i][1] - (0.5 * outputs[0][i][3]),
                    outputs[0][i][2],
                    outputs[0][i][3],
                ]
                boxes.append(box)
                scores.append(maxScore)
                class_ids.append(maxClassIndex)

        # Apply NMS (Non-maximum suppression)
        result_boxes = cv2.dnn.NMSBoxes(boxes, scores, conf, iou, self.s_nms)
        detections = []
        for i in range(len(result_boxes)):
            box = boxes[i]
            box_center = (round(box[0]+box[2]/2), round(box[1]+box[3]/2))
            center_x_cm, center_y_cm, depth_cm = self.intrinsic(box_center, depth, self.rgb_ints)
            index = result_boxes[i]
            name = self.CLASSES[class_ids[index]]
            confidence = scores[index]
            detections.append([center_x_cm, center_y_cm, depth_cm, name, f"{confidence:.2f}"])
            
        detection_strings = []

        for detection in detections:
            name = detection[3]
            x = detection[0]
            y = detection[1]
            d = detection[2]
            c = detection[4]
            detection_strings.append(f"{name} in ({x}, {y}, {d}) with conf: {c}")

        print(", ".join(detection_strings) + "----------------------------------------------")
            
        return detections
        
