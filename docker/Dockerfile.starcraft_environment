FROM coach-base:master as builder

# prep pysc2 and any of its related requirements.
RUN wget http://blzdistsc2-a.akamaihd.net/Linux/SC2.3.17.zip -O sc2.zip \
    && unzip -P 'iagreetotheeula' -d ~ sc2.zip \
    && rm sc2.zip
RUN wget https://github.com/deepmind/pysc2/releases/download/v1.2/mini_games.zip -O mini_games.zip \
    && unzip -d ~/StarCraftII/Maps mini_games.zip \
    && rm mini_games.zip
RUN pip3 install pysc2

# add coach source starting with files that could trigger
# re-build if dependencies change.
RUN mkdir /root/src
COPY setup.py /root/src/.
COPY requirements.txt /root/src/.
RUN pip3 install -r /root/src/requirements.txt

FROM coach-base:master
WORKDIR /root/src
COPY --from=builder /root/StarCraftII /root/StarCraftII
COPY --from=builder /root/.cache /root/.cache
COPY setup.py /root/src/.
COPY requirements.txt /root/src/.
COPY README.md /root/src/.
RUN pip3 install pysc2 && pip3 install -e .[all] && rm -rf /root/.cache
COPY . /root/src
