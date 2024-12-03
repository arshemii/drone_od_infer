# Object Detection on edge
This repository presents a complete, flexible, and ready-to-use application to run object detection models on edge devices (Tested on LattePanda Delta 3, Webcam, and Intel RealSense D435)

--------------------------------------
Detail description:

This repository includes the optimized models of YOLOv8n, YOLOv8s, YOLOv11n, YOLOv11s, and CenterNet dval0 (Updating). The optimization is done using OpenVino NNCF and can be done using [this repository](https://github.com/arshemii/detection_quantization).

After optimizing the model, an optimized inference shall be presented which this repository uses OpenVino runtime since the target hardware is a LattePanda Delta 3 which only has intel CPU and an integrated graphic processor which is not used. Considering that models are optimized or downloaded from this repository, the following order is to run inference:

1. Push the docker image from [here](https://hub.docker.com/repository/docker/arshemii/drone_od/general). Use the latest push command
2. Run the the docker using commands inside [this .sh file](https://github.com/arshemii/drone_od_infer/blob/main/docker_run.sh)
2.1. optional: To ensure using the last repository, remove directory edge_od, and clone this repository again
3. Move to the directory of the repository using cd
4. Run 'python3 main.py' using the flag guide above:
   4.1. --model (example: yolov8n_quantized)
   4.2. --vis (video or text): For webcam use and specifies if you need visualization or textual report
   4.3. --capture (webcam or realsense): Select the device
   4.4. --conf: confidence score
   4.5. --iou: IOU value

   

References:
1. https://github.com/ultralytics
2. https://github.com/openvinotoolkit/openvino

# Contact: arshemii1373@gmail.com
