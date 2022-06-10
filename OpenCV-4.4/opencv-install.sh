dpkg -l | grep "opencv"
echo "The default OpenCV packages are removing (libopencv libopencv-dev libopencv-python libopencv-samples opencv-licenses)"
sudo apt-get remove -y libopencv libopencv-dev libopencv-python libopencv-samples opencv-licenses
sudo apt autoremove -y
sudo apt clean

# Repository setup
sudo apt-add-repository universe
sudo apt-get update

sudo apt-get install -y \
    build-essential \
    cmake \
    libavcodec-dev libavformat-dev libavutil-dev libavresample-dev libdc1394-22-dev \
    libeigen3-dev \
    libglew-dev \
    libgtk2.0-dev libgtk-3-dev \
    libjpeg-dev libpng-dev \
    libpostproc-dev \
    libswscale-dev \
    libtbb-dev \
    libtiff5-dev libtiff-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    qt5-default \
    zlib1g-dev \
    libhdf5-dev \
    libvorbis-dev libxine2-dev \
    libfaac-dev libmp3lame-dev libtheora-dev \
    libopencore-amrnb-dev libopencore-amrwb-dev \
    libopenblas-dev libatlas-base-dev libblas-dev \
    liblapack-dev gfortran \
    libcanberra-gtk* \
    python3-dev python3-numpy python3-py python3-pytest \
    python-dev  python-numpy  python-py  python-pytest \
    v4l-utils \
    protobuf-compiler libprotobuf-dev libgoogle-glog-dev libgflags-dev \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
    pkg-config


sudo apt install ./libopencv-4.4.0_arm64_jetpack_4.6_xavier.deb

# python3 ./opencv_python3_tracker.py
