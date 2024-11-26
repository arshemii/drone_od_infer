#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 15:53:22 2024

@author: arash
"""

from infer_utils import FlexInfer
import argparse
import openvino as ov
import time
import cv2

def main(model, vis, capture, conf, iou):
    flex = FlexInfer()
    
    cmodel = flex.cmodel(model)
    
    a_infer = cmodel.create_infer_request()
    b_infer = cmodel.create_infer_request()

    a_infer.share_inputs = True
    a_infer.share_outputs = True
    b_infer.share_inputs = True
    b_infer.share_outputs = True
    
    input_layer = cmodel.input(0)
    input_shape = cmodel.input(0).shape
    input_tuple = (input_shape[0], input_shape[1], input_shape[2], input_shape[3])
    
    
    if capture == "webcam":
        cap = flex.WC()
        _, frame = cap.read(0)
        frame = flex.wc_preprocess(frame, input_tuple)
        
    if capture == "realsense":
        pipe, align, depth_scale = flex.RS()
        frameset = pipe.wait_for_frames()
        aligned_frames = align.process(frameset)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        frame, depth = flex.rs_preprocess(color_frame, depth_frame, input_tuple, depth_scale)
        
    async_fps = 0
    a_infer.set_tensor(input_layer, ov.Tensor(frame))
    a_infer.start_async()
    frame = 0;
    ti = time.time()
    while True:
        if capture == "webcam":
            _, frame_next = cap.read()
            frame_next = flex.wc_preprocess(frame_next, input_tuple)
            
        if capture == "realsense":
            frameset = pipe.wait_for_frames()
            aligned_frames = align.process(frameset)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            frame_next, depth_next = flex.rs_preprocess(color_frame, depth_frame, input_tuple)
            
        b_infer.set_tensor(input_layer, ov.Tensor(frame_next))
        b_infer.start_async()
        a_infer.wait()
        res = a_infer.get_output_tensor(0).data
        
        total_time = time.time() - ti
        frame = frame + 1
        async_fps = frame / total_time
        print(f"Average FPS: {async_fps}")
        
        if capture == "webcam":
            if vis == "video":
                # Mode is object detection visualization
                frame_next = flex.od_vis(frame_next, res, conf, iou)
                cv2.imshow("image", frame_next)
                key = cv2.waitKey(1)
                # escape = 27
                if key == 27:
                    break
            if vis == "text":
                # Mode is object detection with no visualization
                detections = flex.od_report(res, conf, iou)
        if capture == "realsense":
            # Mode is pose estimation (Report)
            detections = flex.pose_est(res, depth_next, conf, iou, mapping)
            
        frame = frame_next
        a_infer, b_infer = b_infer, a_infer



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Object Detection with OpenVINO")
    
    # Model selection
    parser.add_argument(
        "--model", 
        default="yolov8n_quantized", 
        help="Input your OV model path or name."
    )

    
    
    parser.add_argument(
        "--vis", 
        default="video", 
        choices=["video", "text"], 
        help="Choose if you need visualization or textual report (default: video)."
    )
    
    # Capture source
    parser.add_argument(
        "--capture", 
        default="webcam", 
        choices=["webcam", "realsense"], 
        help="Choose between webcam or Intel RealSense as the capture source (default: webcam)."
    )
    
    
    # Confidence Score
    parser.add_argument(
        "--conf", 
        type=float, 
        default=0.25, 
        help="Confidence score threshold for detections (default: 0.25)."
    )
    
    # IoU
    parser.add_argument(
        "--iou", 
        type=float, 
        default=0.45, 
        help="Minimum IoU threshold for non-maximum suppression (default: 0.45)."
    )
    
    args = parser.parse_args()
    main(args.model, args.vis, args.capture, args.conf, args.iou)
