  docker run -it --rm \
  --user root \
  --device /dev/video0:/dev/video0 \  # Enable access to webcam
  # add other devices (If use Realsense D435, you need all available devices)
  -e QT_QPA_PLATFORM=xcb \
  -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY \
  --cpus="2.0" \     # Adjust it if you need a different limitaion
  --memory="2g" \    # Adjust it if you need a different limitaion
  arshemii/drone_od:26nov24 # Check with the tag of image
