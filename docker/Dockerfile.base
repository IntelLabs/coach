FROM nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04

################################
# Install apt-get Requirements #
################################

# General
RUN apt-get update && \
    apt-get install -y python3-pip cmake zlib1g-dev python3-tk python-opencv \
    # Boost libraries
    libboost-all-dev \
    # Scipy requirements
    libblas-dev liblapack-dev libatlas-base-dev gfortran \
    # Pygame requirements
    libsdl-dev libsdl-image1.2-dev libsdl-mixer1.2-dev libsdl-ttf2.0-dev \
    libsmpeg-dev libportmidi-dev libavformat-dev libswscale-dev \
    # Dashboard
    dpkg-dev build-essential python3.5-dev libjpeg-dev  libtiff-dev libsdl1.2-dev libnotify-dev \
    freeglut3 freeglut3-dev libsm-dev libgtk2.0-dev libgtk-3-dev libwebkitgtk-dev libgtk-3-dev \
    libwebkitgtk-3.0-dev libgstreamer-plugins-base1.0-dev \
    # Gym
    libav-tools libsdl2-dev swig cmake \
    # Mujoco_py
    curl libgl1-mesa-dev libgl1-mesa-glx libglew-dev libosmesa6-dev software-properties-common \
    # ViZDoom
    build-essential zlib1g-dev libsdl2-dev libjpeg-dev \
    nasm tar libbz2-dev libgtk2.0-dev cmake git libfluidsynth-dev libgme-dev \
    libopenal-dev timidity libwildmidi-dev unzip wget && \
    apt-get clean autoclean && \
    apt-get autoremove -y

############################
# Install Pip Requirements #
############################
RUN pip3 install --upgrade pip
RUN pip3 install setuptools==39.1.0 && pip3 install pytest && pip3 install pytest-xdist

RUN curl -o /usr/local/bin/patchelf https://s3-us-west-2.amazonaws.com/openai-sci-artifacts/manual-builds/patchelf_0.9_amd64.elf \
    && chmod +x /usr/local/bin/patchelf
