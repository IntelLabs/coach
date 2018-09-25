# REGISTRY=nervana-dockrepo01.fm.intel.com:5001/
# REGISTRY=gcr.io/
REGISTRY=docker.io/
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

unit_tests: build
	${DOCKER} run ${RUN_ARGUMENTS} -it ${IMAGE} python3 -m pytest rl_coach/tests -m unit_test -n 8

integration_tests: build
	${DOCKER} run ${RUN_ARGUMENTS} -it ${IMAGE} python3 -m pytest rl_coach/tests -m integration_test -n auto

golden_tests: build
	${DOCKER} run ${RUN_ARGUMENTS} -it ${IMAGE} python3 rl_coach/tests/golden_tests.py

trace_tests: build
	${DOCKER} run ${RUN_ARGUMENTS} -it ${IMAGE} python3 rl_coach/tests/trace_tests.py -prl

run: build
	${DOCKER} run ${RUN_ARGUMENTS} -it ${IMAGE}

run_training_worker: build
	${DOCKER} run ${RUN_ARGUMENTS} -it ${IMAGE} python3 rl_coach/training_worker.py --preset CartPole_DQN_distributed

run_rollout_worker: build
	${DOCKER} run ${RUN_ARGUMENTS} -it ${IMAGE} python3 rl_coach/rollout_worker.py --preset CartPole_DQN_distributed

bootstrap_kubernetes: build push
	kubectl run -i --tty --attach --image=${REGISTRY}${IMAGE} --restart=Never distributed-coach -- python3 rl_coach/orchestrators/start_training.py --preset CartPole_DQN_distributed --image ${IMAGE} -ns 10.63.249.182 -np /

stop_kubernetes:
	kubectl delete service --ignore-not-found redis-service
	kubectl delete pv --ignore-not-found nfs-checkpoint-pv
	kubectl delete pvc --ignore-not-found nfs-checkpoint-pvc
	kubectl delete deployment --ignore-not-found redis-server
	kubectl get jobs | grep train | awk "{print $\1}" | xargs kubectl delete jobs
	kubectl get jobs | grep worker | awk "{print $\1}" | xargs kubectl delete jobs

kubernetes: stop_kubernetes
	python3 ${CONTEXT}/rl_coach/orchestrators/start_training.py --preset CartPole_DQN_distributed --image ${IMAGE} -ns 10.63.249.182 -np /

push: build
	${DOCKER} tag ${IMAGE} ${REGISTRY}${IMAGE}
	${DOCKER} push ${REGISTRY}${IMAGE}
