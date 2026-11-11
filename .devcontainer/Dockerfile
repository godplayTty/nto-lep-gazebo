FROM ros:jazzy-ros-base-noble
RUN apt-get update \
    && apt-get install -y --no-install-recommends ros-jazzy-ros-gz ros-jazzy-tf2-ros \
       python3-opencv python3-numpy libgl1-mesa-dri libegl-mesa0 \
    && rm -rf /var/lib/apt/lists/*
ENV LIBGL_ALWAYS_SOFTWARE=1
