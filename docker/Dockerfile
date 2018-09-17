FROM ubuntu:16.04

RUN apt-get update \
    && apt-get install -y \
    python3-pip cmake zlib1g-dev python3-tk python-opencv \
    libboost-all-dev \
    libblas-dev liblapack-dev libatlas-base-dev gfortran \
    libsdl-dev libsdl-image1.2-dev libsdl-mixer1.2-dev libsdl-ttf2.0-dev \
    libsmpeg-dev libportmidi-dev libavformat-dev libswscale-dev \
    dpkg-dev build-essential python3.5-dev libjpeg-dev  libtiff-dev \
    libsdl1.2-dev libnotify-dev freeglut3 freeglut3-dev libsm-dev \
    libgtk2.0-dev libgtk-3-dev libwebkitgtk-dev libgtk-3-dev \
    libwebkitgtk-3.0-dev libgstreamer-plugins-base1.0-dev \
    libav-tools libsdl2-dev swig

# installing python dependencies
RUN pip3 install --upgrade pip

RUN mkdir /root/src
COPY setup.py /root/src/.
COPY requirements.txt /root/src/.
COPY README.md /root/src/.
WORKDIR /root/src
RUN pip3 install -e .
#
COPY . /root/src
WORKDIR /root/src
RUN pip3 install -e .

# RUN pip3 install rl_coach

# CMD ["coach", "-p", "CartPole_PG", "-e", "cartpole"]
# CMD python3 rl_coach/rollout_worker.py --preset CartPole_DQN_distributed
