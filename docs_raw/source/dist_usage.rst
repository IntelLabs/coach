.. _dist-coach-usage:

Usage - Distributed Coach
=========================

Coach supports the horizontal scale-out of rollout workers in distributed mode. For more information on the design and
implementation of distributed Coach, see :ref:`dist-coach-design`. In the rest of this section, we will describe how to
get started with distributed Coach.

Interfaces and Implementations
------------------------------

Coach uses three interfaces to orchestrate, schedule and manager the resources of workers it spawns in the distributed
mode. These interfaces are the orchestrator, memory backend and the data store. Refer to :ref:`dist-coach-design` for
more information. The following implementation(s) are available for each interface:

* **Orchestrator** - `Kubernetes <https://kubernetes.io>`_.
* **Memory Backend** - `Redis Pub/Sub <https://redis.io/topics/pubsub>`_.
* **Data Store** - `S3 <https://aws.amazon.com/s3>`_ and `NFS <https://en.wikipedia.org/wiki/Network_File_System>`_.

Prerequisites
-------------

* Building and pushing containers - `Docker <https://docs.docker.com/install/linux/docker-ce/ubuntu>`_.
* Container registry access for hosting container images - `Docker Hub <https://hub.docker.com>`_
* Using Kubernetes for orchestration - `Kubernetes configuration <https://kubernetes.io/docs/tasks/access-application-cluster/configure-access-multiple-clusters/>`_.
* Using S3 for storing policy checkpoints - `AWS CLI <https://docs.aws.amazon.com/cli/latest/userguide/installing.html>_,
  `AWS credentials <https://aws.amazon.com/blogs/security/a-new-and-standardized-way-to-manage-credentials-in-the-aws-sdks>`_
  and `S3 bucket <https://docs.aws.amazon.com/AmazonS3/latest/user-guide/create-bucket.html>`_.

Clone the Repository
--------------------

.. code-block:: bash

   $ git clone git@github.com:NervanaSystems/coach.git
   $ cd coach

Build Container Image and Push
------------------------------
Create a directory `docker`.

.. code-block:: bash

   $ mkdir docker

Create docker files in the `docker` directory.

A sample base docker file (Dockerfile.base) would look like this:

.. code-block:: bash

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


A sample docker file for the gym environment would look like this:

.. code-block:: bash

   FROM coach-base:master as builder

   # prep gym and any of its related requirements.
   RUN pip3 install gym[atari,box2d,classic_control]==0.10.5

   # add coach source starting with files that could trigger
   # re-build if dependencies change.
   RUN mkdir /root/src
   COPY setup.py /root/src/.
   COPY requirements.txt /root/src/.
   RUN pip3 install -r /root/src/requirements.txt

   FROM coach-base:master
   WORKDIR /root/src
   COPY --from=builder /root/.cache /root/.cache
   COPY setup.py /root/src/.
   COPY requirements.txt /root/src/.
   COPY README.md /root/src/.
   RUN pip3 install gym[atari,box2d,classic_control]==0.10.5 && pip3 install -e .[all] && rm -rf /root/.cache
   COPY . /root/src


A sample docker file for the Mujoco environment would look like this:

.. code-block:: bash

   FROM coach-base:master as builder

   # prep mujoco and any of its related requirements.
   # Mujoco
   RUN mkdir -p ~/.mujoco \
       && wget https://www.roboti.us/download/mjpro150_linux.zip -O mujoco.zip \
       && unzip -n mujoco.zip -d ~/.mujoco \
       && rm mujoco.zip
   ARG MUJOCO_KEY
   ENV MUJOCO_KEY=$MUJOCO_KEY
   ENV LD_LIBRARY_PATH /root/.mujoco/mjpro150/bin:$LD_LIBRARY_PATH
   RUN echo $MUJOCO_KEY | base64 --decode > /root/.mujoco/mjkey.txt
   RUN pip3 install mujoco_py

   # add coach source starting with files that could trigger
   # re-build if dependencies change.
   RUN mkdir /root/src
   COPY setup.py /root/src/.
   COPY requirements.txt /root/src/.
   RUN pip3 install -r /root/src/requirements.txt

   FROM coach-base:master
   WORKDIR /root/src
   COPY --from=builder /root/.mujoco /root/.mujoco
   ENV LD_LIBRARY_PATH /root/.mujoco/mjpro150/bin:$LD_LIBRARY_PATH
   COPY --from=builder /root/.cache /root/.cache
   COPY setup.py /root/src/.
   COPY requirements.txt /root/src/.
   COPY README.md /root/src/.
   RUN pip3 install mujoco_py && pip3 install -e .[all] && rm -rf /root/.cache
   COPY . /root/src


A sample docker file for the ViZDoom environment would look like this:

.. code-block:: bash

   FROM coach-base:master as builder
   
   # prep vizdoom and any of its related requirements.
   RUN pip3 install vizdoom
   
   # add coach source starting with files that could trigger
   # re-build if dependencies change.
   RUN mkdir /root/src
   COPY setup.py /root/src/.
   COPY requirements.txt /root/src/.
   RUN pip3 install -r /root/src/requirements.txt
   
   FROM coach-base:master
   WORKDIR /root/src
   COPY --from=builder /root/.cache /root/.cache
   COPY setup.py /root/src/.
   COPY requirements.txt /root/src/.
   COPY README.md /root/src/.
   RUN pip3 install vizdoom && pip3 install -e .[all] && rm -rf /root/.cache
   COPY . /root/src



Build the base container. Make sure you are in the Coach root directory before building.

.. code-block:: bash

   $ docker build -t coach-base:master -f docker/Dockerfile.base .

If you would like to use the Mujoco environment, save this key as an environment variable. Replace `<mujoco_key>` with the
contents of your mujoco key.

.. code-block:: bash

   $ export MUJOCO_KEY=<mujoco_key>

Build the container for your environment.
Replace `<env>` with your choice of environment. The choices are `gym`, `mujoco` and `doom`.
Replace `<user-name>`, `<image-name>` and `<tag>` with appropriate values.

.. code-block:: bash

   $ docker build --build-arg MUJOCO_KEY=${MUJOCO_KEY} -t <user-name>/<image-name>:<tag> -f docker/Dockerfile.<env> .

Push the container to a registry of your choice. Replace `<user-name>`, `<image-name>` and `<tag>` with appropriate values.

.. code-block:: bash

   $ docker push <user-name>/<image-name>:<tag>

Create a Config file
--------------------

Add the following contents to file.
Replace `<user-name>`, `<image-name>`, `<tag>`, `<bucket-name>` and `<path-to-aws-credentials>` with appropriate values.

.. code-block:: bash

   [coach]
   image = <user-name>/<image-name>:<tag>
   memory_backend = redispubsub
   data_store = s3
   s3_end_point = s3.amazonaws.com
   s3_bucket_name = <bucket-name>
   s3_creds_file = <path-to-aws-credentials>

Run Distributed Coach
---------------------

The following command will run distributed Coach with CartPole_ClippedPPO preset, Redis Pub/Sub as the memory backend, S3 as the data store in Kubernetes
with three rollout workers.

.. code-block:: bash

   $ python3 rl_coach/coach.py -p CartPole_ClippedPPO \
   -dc \
   -e <experiment-name> \
   -n 3 \
   -dcp <path-to-config-file>
