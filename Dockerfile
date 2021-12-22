FROM python:3.6.15-bullseye

RUN echo 'root:root' | chpasswd
ENV SSH_AUTH_SOCK=/ssh-agent

# --- Create non-root user with the ability to use sudo --- #
ARG USERNAME=developer
ARG USER_UID=20141
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    #
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

WORKDIR /home/developer/

# In order for Doom to work
RUN sudo apt -y install cmake libboost-all-dev libsdl2-dev libfreetype6-dev libgl1-mesa-dev libglu1-mesa-dev libpng-dev libjpeg-dev libbz2-dev libfluidsynth-dev libgme-dev libopenal-dev zlib1g-dev timidity tar nasm

# General
RUN sudo -E apt-get install python3-pip cmake zlib1g-dev python3-tk python3-opencv -y

# Boost libraries
RUN sudo -E apt-get install libboost-all-dev -y

# Scipy requirements
RUN sudo -E apt-get install libblas-dev liblapack-dev libatlas-base-dev gfortran -y

# PyGame
RUN sudo -E apt-get install libsdl-dev libsdl-image1.2-dev libsdl-mixer1.2-dev libsdl-ttf2.0-dev libsmpeg-dev libportmidi-dev libavformat-dev libswscale-dev -y

USER $USERNAME

# # To create and change context
# docker context rm DFP_development
# # $1 Is the external IP adress
# docker context create DFP_development --docker "host=ssh://theo.vincent@$1"
# docker context use DFP_development

# docker run -d -it -v /home/theo.vincent/:/home/developer/ gcr.io/directfutureprediction/development_vm:latest bash