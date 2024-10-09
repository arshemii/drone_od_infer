# edge_od
This repository presents necessary scripts and files for a ready-to-use object detection on a docker container with CPU and Intel Realsense camera.

To use, follow the steps:

# Pull the docker ----------------------------------------------------

Inside your terminal run the following command:

docker pull arshemii/edge_od:091024



# Open a container --------------------------------------------------

Download docker_run.sh file and run it, or use the following commands in your CLI:

#!/bin/bash

xhost +local:docker && sudo docker run -e DISPLAY=$DISPLAY \
    -e QT_QPA_PLATFORM=xcb \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    --cpus="2" \
    -v ~/.Xauthority:/root/.Xauthority \
    --device /dev/video2 \
    --device /dev/video3 \
    --device /dev/video4 \
    --device /dev/video5 \
    --device /dev/video6 \
    --device /dev/video7 \
    --device /dev/video8 \
    -it --ipc=host drone_od:091024 bash



# Inter-container update

To use the last version of this repository, once you open a container, run the following commands in a terminal inside your container:

cd ultralytics
chmod +x cont_update.sh
./cont_update.sh

*** It will remove the previous repository and replace it with the new one, so you need an internet connection.



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
