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
