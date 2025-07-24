# Kalfilter.py

This file contain all the methods and definition for the kalman filter class

# MotionDetector.py

This file contain all the methods needed to draw bbox, update frames, use kalman filter to track objects, assigned, and find the blobs region in the motion image

updated will check if we have 3 frame to make motion and create the objects list or not, will have 1 frame delay, skip is check before calling update_objects, match each object with it closet detection of bbox, basically finding it previous location to match to that again

# tracking.py

sample from qtdemo.py on github + adding skip frame and backward 60 frames

have a buffer to prevent motion breakout when forward or backward.

# to run

- have uv.lock or uv installed
- uv sync
- uv run .\tracking.py .\east_parking_reduced_size.mp4
  or python .\tracking.py .\east_parking_reduced_size.mp4
