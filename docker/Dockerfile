FROM coach-base:master

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
