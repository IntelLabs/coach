REGISTRY=nervana-dockrepo01.fm.intel.com:5001/
IMAGE=zdwiel/coach
# IMAGE=gcr.io/deep-greens/inference:v5

BUILD_ARGUMENTS=
RUN_ARGUMENTS=
ifdef http_proxy
	BUILD_ARGUMENTS+=--build-arg http_proxy=$(http_proxy)
	RUN_ARGUMENTS+=--env http_proxy=$(http_proxy)
endif

ifdef https_proxy
	BUILD_ARGUMENTS+=--build-arg https_proxy=$(https_proxy)
	RUN_ARGUMENTS+=--env https_proxy=$(https_proxy)
endif

RUN_ARGUMENTS+=--rm
RUN_ARGUMENTS+=--net host
RUN_ARGUMENTS+=-v /tmp/checkpoint:/checkpoint

CONTEXT = $(realpath ..)

ifndef DOCKER
	DOCKER = docker
endif

build:
	${DOCKER} build -f=Dockerfile -t=${IMAGE} ${BUILD_ARGUMENTS} ${CONTEXT}
	mkdir -p /tmp/checkpoint
	rm -rf /tmp/checkpoint/*

shell: build
	${DOCKER} run ${RUN_ARGUMENTS} -it ${IMAGE} /bin/bash

test: build
	${DOCKER} run ${RUN_ARGUMENTS} -it ${IMAGE} py.test tests

run: build
	${DOCKER} run ${RUN_ARGUMENTS} -it ${IMAGE}

run_training_worker: build
	${DOCKER} run ${RUN_ARGUMENTS} -it ${IMAGE} python3 rl_coach/training_worker.py --preset CartPole_DQN_distributed

run_rollout_worker: build
	${DOCKER} run ${RUN_ARGUMENTS} -it ${IMAGE} python3 rl_coach/rollout_worker.py --preset CartPole_DQN_distributed

kubernetes: build push
	kubectl run -i --tty --attach --image=${REGISTRY}${IMAGE} --restart=Never distributed-coach -- python3 rl_coach/orchestrators/start_training.py --preset CartPole_DQN_distributed --image ${IMAGE}

push: build
	${DOCKER} tag ${IMAGE} ${REGISTRY}${IMAGE}
	${DOCKER} push ${REGISTRY}${IMAGE}
