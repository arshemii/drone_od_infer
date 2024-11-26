# YOLO on edge
This repository presents a complete, flexible, and ready-to-use application to run YOLO on edge devices (Tested on LattePanda Delta 3, Webcam, and Intel RealSense D435)

--------------------------------------
Detail description:

This repository includes the optimized models of YOLOv8n, YOLOv8s, YOLOv11n, and YOLOv11s (Updating). The optimization is done using OpenVino NNCF and can be done using [this repository](https://github.com/arshemii/yolo_optimization).

After optimizing the model, an optimized inference shall be presented which this repository uses OpenVino runtime since the target hardware is a LattePanda Delta 3 which only has intel CPU and an integrated graphic processor which is not used. Considering that models are optimized or downloaded from this repository, the following order is to run inference:

1. Push the docker image from [here](https://hub.docker.com/repository/docker/arshemii/drone_od/general). Use the latest push command
2. Run the the docker using commands inside [this .sh file](https://github.com/arshemii/drone_od_infer/blob/main/docker_run.sh)
2.1. optional: To ensure using the last repository, remove directory edge_od, and clone this repository again
3. Move to the directory of the repository using cd
4. Run 'python3 main.py' using the flag guide above:
   4.1. 




# Run object detection

Go to the repository directory using:
cd edge_od
and run:
main.py

# Important

1. The docker image is a simplified and light version of ultralytics for CPU use
2. You can change python files for more personalization using nano text editor inside a container.
3. You might need to change the devices for video while opening a container
4. You can add other models for inference inside your cloned repository


# For more information: a.hashemi@studenti.unina.it
