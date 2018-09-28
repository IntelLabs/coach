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

################################
# Install apt-get Requirements #
################################

# General
RUN apt-get update && \
    apt-get install -y python3-pip cmake zlib1g-dev python3-tk python-opencv && \
    apt-get clean autoclean && \
    apt-get autoremove -y

# Boost libraries
RUN apt-get update && \
    apt-get install -y libboost-all-dev && \
    apt-get clean autoclean && \
    apt-get autoremove -y

# Scipy requirements
RUN apt-get update && \
    apt-get install -y libblas-dev liblapack-dev libatlas-base-dev gfortran && \
    apt-get clean autoclean && \
    apt-get autoremove -y

# Pygame requirements
RUN apt-get update && \
    apt-get install -y libsdl-dev libsdl-image1.2-dev libsdl-mixer1.2-dev libsdl-ttf2.0-dev && \
    apt-get clean autoclean && \
    apt-get autoremove -y
RUN apt-get update && \
    apt-get install -y libsmpeg-dev libportmidi-dev libavformat-dev libswscale-dev && \
    apt-get clean autoclean && \
    apt-get autoremove -y

# Dashboard
RUN apt-get update && \
    apt-get install -y dpkg-dev build-essential python3.5-dev libjpeg-dev  libtiff-dev libsdl1.2-dev libnotify-dev \
    freeglut3 freeglut3-dev libsm-dev libgtk2.0-dev libgtk-3-dev libwebkitgtk-dev libgtk-3-dev \
    libwebkitgtk-3.0-dev libgstreamer-plugins-base1.0-dev && \
    apt-get clean autoclean && \
    apt-get autoremove -y

# Gym
RUN apt-get update && \
    apt-get install -y libav-tools libsdl2-dev swig cmake && \
    apt-get clean autoclean && \
    apt-get autoremove -y

# Mujoco_py
RUN apt-get update && \
    apt-get install -y curl libgl1-mesa-dev libgl1-mesa-glx libglew-dev libosmesa6-dev software-properties-common && \
    apt-get clean autoclean && \
    apt-get autoremove -y

# ViZDoom
RUN apt-get update && \
    apt-get install -y build-essential zlib1g-dev libsdl2-dev libjpeg-dev \
    nasm tar libbz2-dev libgtk2.0-dev cmake git libfluidsynth-dev libgme-dev \
    libopenal-dev timidity libwildmidi-dev unzip wget && \
    apt-get clean autoclean && \
    apt-get autoremove -y

############################
# Install Pip Requirements #
############################
RUN pip3 install --upgrade pip
RUN pip3 install pytest
RUN pip3 install pytest-xdist

# initial installation of coach, so that the docker build won't install everything from scratch
RUN pip3 install rl_coach>=0.10.0

# install additional environments
RUN pip3 install gym[atari]==0.10.5
RUN pip3 install mujoco_py==1.50.1.56
RUN pip3 install vizdoom==1.1.6

# FROM ubuntu:16.04
#
# RUN apt-get update \
#     && apt-get install -y \
#     python3-pip cmake zlib1g-dev python3-tk python-opencv \
#     libboost-all-dev \
#     libblas-dev liblapack-dev libatlas-base-dev gfortran \
#     libsdl-dev libsdl-image1.2-dev libsdl-mixer1.2-dev libsdl-ttf2.0-dev \
#     libsmpeg-dev libportmidi-dev libavformat-dev libswscale-dev \
#     dpkg-dev build-essential python3.5-dev libjpeg-dev  libtiff-dev \
#     libsdl1.2-dev libnotify-dev freeglut3 freeglut3-dev libsm-dev \
#     libgtk2.0-dev libgtk-3-dev libwebkitgtk-dev libgtk-3-dev \
#     libwebkitgtk-3.0-dev libgstreamer-plugins-base1.0-dev \
#     libav-tools libsdl2-dev swig
#
# # installing python dependencies
# RUN pip3 install --upgrade pip

RUN apt-get update && apt-get install -y wget zip
RUN mkdir -p ~/.mujoco \
    && wget https://www.roboti.us/download/mjpro150_linux.zip -O mujoco.zip \
    && unzip mujoco.zip -d ~/.mujoco \
    && rm mujoco.zip
COPY ./mjkey.txt /root/.mujoco/
ENV LD_LIBRARY_PATH /root/.mujoco/mjpro150/bin:$LD_LIBRARY_PATH

RUN curl -o /usr/local/bin/patchelf https://s3-us-west-2.amazonaws.com/openai-sci-artifacts/manual-builds/patchelf_0.9_amd64.elf \
    && chmod +x /usr/local/bin/patchelf

RUN mkdir /root/src
COPY setup.py /root/src/.
COPY requirements.txt /root/src/.
COPY README.md /root/src/.
WORKDIR /root/src
RUN pip3 install -e .

# everything above here should be cached most of the time
COPY . /root/src
WORKDIR /root/src
RUN pip3 install -e .

RUN chmod 777 /root/src/docker/docker_entrypoint.sh
ENTRYPOINT ["/root/src/docker/docker_entrypoint.sh"]
