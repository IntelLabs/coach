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
