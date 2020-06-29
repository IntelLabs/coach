FROM coach-base:master as builder

# prep some of the more common environments
# Gym (installed with coach)
Run pip3 install gym[atari]==0.12.5 box2d
# Mujoco
RUN mkdir -p ~/.mujoco \
 && wget https://www.roboti.us/download/mujoco200_linux.zip -O mujoco.zip \
 && unzip mujoco.zip -d ~/.mujoco \
 && mv /root/.mujoco/mujoco200_linux /root/.mujoco/mujoco200 \
 && rm mujoco.zip
ARG MUJOCO_KEY
ENV MUJOCO_KEY=$MUJOCO_KEY
RUN echo $MUJOCO_KEY | base64 --decode > /root/.mujoco/mjkey.txt
ENV LD_LIBRARY_PATH /root/.mujoco/mujoco200/bin:${LD_LIBRARY_PATH}
RUN pip3 install cffi
RUN pip3 install Cython
RUN pip3 install mujoco-py>=2.0

# Vizdoom
RUN pip3 install vizdoom

RUN mkdir /root/src
COPY setup.py /root/src/.
COPY requirements.txt /root/src/.
RUN pip3 install -r /root/src/requirements.txt

FROM coach-base:master
WORKDIR /root/src
COPY --from=builder /root/.mujoco /root/.mujoco
COPY --from=builder /root/.cache /root/.cache
COPY setup.py /root/src/.
COPY requirements.txt /root/src/.
COPY README.md /root/src/.
RUN pip3 install gym[atari]==0.12.5 box2d cffi mujoco-py>=2.0 vizdoom && pip3 install -e .[all] && rm -rf /root/.cache
COPY . /root/src
