# REGISTRY=gcr.io
REGISTRY=docker.io
ORGANIZATION=nervana
IMAGE=coach
CONTEXT = $(realpath ..)

BUILD_ARGUMENTS=
RUN_ARGUMENTS=
ifdef http_proxy
	BUILD_ARGUMENTS+=--build-arg http_proxy=$(http_proxy)
endif

ifdef https_proxy
	BUILD_ARGUMENTS+=--build-arg https_proxy=$(https_proxy)
endif

ifndef MUJOCO_KEY
KEY_NAME=${CONTEXT}/mjkey.txt
ifneq ("$(wildcard ${KEY_NAME})", "")
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
		MUJOCO_KEY:=$(shell cat ${KEY_NAME} | base64 -w 0)
endif
ifeq ($(UNAME_S),Darwin)
		MUJOCO_KEY:=$(shell cat ${KEY_NAME} | base64)
endif
endif
endif

ifdef MUJOCO_KEY
	BUILD_ARGUMENTS+=--build-arg MUJOCO_KEY=$(MUJOCO_KEY)
endif

RUN_ARGUMENTS+=--rm
RUN_ARGUMENTS+=--net host
RUN_ARGUMENTS+=-v /tmp/checkpoint:/checkpoint

UNIT_TESTS=python3 -m pytest rl_coach/tests -m unit_test
INTEGRATION_TESTS=python3 -m pytest rl_coach/tests -m integration_test --tb=short

# Functional Tests
ifdef FUNCTIONAL_PRESETS
	FUNCTIONAL_PRESETS:=-k "${FUNCTIONAL_PRESETS}"
endif
ifndef FUNCTIONAL_PRESETS
	FUNCTIONAL_PRESETS=
endif
FUNCTIONAL_TESTS=python3 -m pytest rl_coach/tests -s -v ${FUNCTIONAL_PRESETS} -m functional_test

# Golden Tests
ifdef GOLDEN_PRESETS
	GOLDEN_PRESETS:=-k "${GOLDEN_PRESETS}"
endif
ifndef GOLDEN_PRESETS
	GOLDEN_PRESETS=
endif
GOLDEN_TESTS=python3 -m pytest rl_coach/tests -m golden_test -v ${GOLDEN_PRESETS}
# example specifying a couple presets:
# GOLDEN_TESTS=python3 -m pytest rl_coach/tests -m golden_test -v -k "Mujoco_NAF or Doom_Basic_DQN"
# example specifying all golden tests using the mujoco environment
# GOLDEN_TESTS=python3 -m pytest rl_coach/tests -m golden_test -v -k "Mujoco"

# Trace Tests
ifdef TRACE_PRESETS
	TRACE_PRESETS := -p $(TRACE_PRESETS)
else
	TRACE_PRESETS :=
endif
TRACE_TESTS=python3 rl_coach/tests/trace_tests.py -prl ${TRACE_PRESETS}

ifndef DOCKER
	DOCKER = docker
endif

build_base:
	${DOCKER} build -f=Dockerfile.base -t=${IMAGE}-base:master ${BUILD_ARGUMENTS} ${CONTEXT}

build: build_base
	${DOCKER} build -f=Dockerfile -t=${IMAGE} ${BUILD_ARGUMENTS} ${CONTEXT}
	mkdir -p /tmp/checkpoint
	rm -rf /tmp/checkpoint/*

shell: build
	${DOCKER} run ${RUN_ARGUMENTS} -it ${IMAGE} /bin/bash

unit_tests: build
	${DOCKER} run ${RUN_ARGUMENTS} -it ${IMAGE} ${UNIT_TESTS}

integration_tests: build
	${DOCKER} run ${RUN_ARGUMENTS} -it ${IMAGE} ${INTEGRATION_TESTS}

functional_tests: build
	${DOCKER} run ${RUN_ARGUMENTS} -it ${IMAGE} ${FUNCTIONAL_TESTS}

golden_tests: build
	${DOCKER} run ${RUN_ARGUMENTS} -it ${IMAGE} ${GOLDEN_TESTS}

trace_tests: build
	${DOCKER} run ${RUN_ARGUMENTS} -it ${IMAGE} ${TRACE_TESTS}

run: build
	${DOCKER} run ${RUN_ARGUMENTS} -it ${IMAGE}

run_training_worker: build
	${DOCKER} run ${RUN_ARGUMENTS} -it ${IMAGE} python3 rl_coach/training_worker.py --preset CartPole_DQN_distributed

run_rollout_worker: build
	${DOCKER} run ${RUN_ARGUMENTS} -it ${IMAGE} python3 rl_coach/rollout_worker.py --preset CartPole_DQN_distributed

bootstrap_kubernetes: build push
	kubectl run -i --tty --attach --image=${REGISTRY}/${IMAGE} --restart=Never distributed-coach -- python3 rl_coach/orchestrators/start_training.py --preset CartPole_DQN_distributed --image ${IMAGE} -ns 10.63.249.182 -np /

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
	${DOCKER} tag ${IMAGE} ${REGISTRY}/${ORGANIZATION}/${IMAGE}
	${DOCKER} push ${REGISTRY}/${ORGANIZATION}/${IMAGE}

unit_tests_without_docker:
	cd .. && ${UNIT_TESTS}

integration_tests_without_docker:
	cd .. && ${INTEGRATION_TESTS}

functional_tests_without_docker:
	cd .. && ${FUNCTIONAL_TESTS}

golden_tests_without_docker:
	cd .. && ${GOLDEN_TESTS}

trace_tests_without_docker:
	cd .. && ${TRACE_TESTS}
