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
RUN pip3 install mujoco_py==1.50.1.68

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
RUN pip3 install mujoco_py==1.50.1.68 && pip3 install -e .[all] && rm -rf /root/.cache
COPY . /root/src
