FROM nvidia/cuda:11.2.1-cudnn8-runtime-ubuntu18.04

# Install packages without prompting the user to answer any questions
ENV DEBIAN_FRONTEND noninteractive 

# Install packages
RUN apt-get update && apt-get install -y \
lsb-release \
mesa-utils \
git \
vim \
wget \
curl \
libssl-dev \
build-essential \
cmake \
pkg-config \
clang  \
dbus-x11 \
libomp-dev \       
v4l-utils libv4l-dev \               
libudev-dev \                   
libopencv-dev \              
libjson-c-dev \                 
libpng-dev \                   
libgtk-3-dev &&\
apt-get clean && rm -rf /var/lib/apt/lists/*

# Install ROS
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
RUN apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
RUN apt-get update && apt-get install -y --allow-downgrades --allow-remove-essential --allow-change-held-packages \
libpcap-dev libpcl-dev \
gstreamer1.0-tools libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libgstreamer-plugins-good1.0-dev \
ros-melodic-desktop-full python-rosinstall python-rosinstall-generator python-wstool build-essential python-rosdep \
ros-melodic-socketcan-bridge ros-melodic-ros-numpy python-pip python3-pip \
python3-catkin-pkg-modules \
python-catkin-tools \
ros-melodic-geodesy && \
apt-get clean && rm -rf /var/lib/apt/lists/*

# Configure ROS
RUN rosdep init && rosdep update 

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES \
    ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

RUN addgroup --gid 1000 user
RUN adduser --disabled-password --gecos '' --uid 1000 --gid 1000 user
RUN usermod -a -G video user
RUN mkdir -p /home/user
WORKDIR /home/user
RUN chown -R user /home/user
COPY . /home/user
RUN cd ACSC/segmentation && python3 setup.py install 
RUN pip3 install --upgrade pip && python3 -m pip install checkerboard
RUN pip3 install --upgrade pip && python3 -m pip install PyYAML scipy sklearn transforms3d matplotlib cython
RUN cd python-pcl && python3 setup.py build_ext -i && python3 setup.py install
RUN mkdir /home/user/calibration_data





