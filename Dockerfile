FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

# https://github.com/NVIDIA/nvidia-docker/issues/619
RUN rm /etc/apt/sources.list.d/cuda.list

RUN apt-get update  && \
    apt-get upgrade -y && \
    apt-get clean autoclean && \
    apt-get autoremove -y

RUN apt-get update && \
    apt-get install -y python-pip && \
    apt-get clean autoclean && \
    apt-get autoremove -y
RUN pip install pip --upgrade
WORKDIR /root

# update apt-get
RUN apt-get update && \
    apt-get upgrade -y

################################
# Install apt-get Requirements #
################################

# General
RUN apt-get install python3-pip cmake zlib1g-dev python3-tk python-opencv -y

# Boost libraries
RUN apt-get install libboost-all-dev -y

# Scipy requirements
RUN apt-get install libblas-dev liblapack-dev libatlas-base-dev gfortran -y

# Pygame requirements
RUN apt-get install libsdl-dev libsdl-image1.2-dev libsdl-mixer1.2-dev libsdl-ttf2.0-dev -y
RUN apt-get install libsmpeg-dev libportmidi-dev libavformat-dev libswscale-dev -y

# Dashboard
RUN apt-get install dpkg-dev build-essential python3.5-dev libjpeg-dev  libtiff-dev libsdl1.2-dev libnotify-dev \
        freeglut3 freeglut3-dev libsm-dev libgtk2.0-dev libgtk-3-dev libwebkitgtk-dev libgtk-3-dev \
        libwebkitgtk-3.0-dev libgstreamer-plugins-base1.0-dev -y

# Gym
RUN apt-get install libav-tools libsdl2-dev swig cmake -y

# Mujoco_py
RUN apt-get install curl libgl1-mesa-dev libgl1-mesa-glx libglew-dev libosmesa6-dev software-properties-common -y

# ViZDoom
RUN apt-get install build-essential zlib1g-dev libsdl2-dev libjpeg-dev \
    nasm tar libbz2-dev libgtk2.0-dev cmake git libfluidsynth-dev libgme-dev \
    libopenal-dev timidity libwildmidi-dev unzip wget -y
    
# cleanup apt-get
RUN apt-get clean autoclean && \
    apt-get autoremove -y

############################
# Install Pip Requirements #
############################

RUN pip3 install --upgrade pip

RUN pip3 install pytest

# initial installation of coach, so that the docker build won't install everything from scratch
RUN pip3 install rl_coach>=0.10.0

# install additional environments
RUN pip3 install gym[atari]==0.10.5
RUN pip3 install mujoco_py==1.50.1.56
RUN pip3 install vizdoom==1.1.6

COPY . /root/src
WORKDIR /root/src

RUN pip3 install -e .

RUN curl -o /usr/local/bin/patchelf https://s3-us-west-2.amazonaws.com/openai-sci-artifacts/manual-builds/patchelf_0.9_amd64.elf \
    && chmod +x /usr/local/bin/patchelf

RUN chmod 777 /root/src/docker_entrypoint
ENTRYPOINT ["/root/src/docker_entrypoint"]
