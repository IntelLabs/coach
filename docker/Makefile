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

CONTEXT = $(realpath ..)

ifndef DOCKER
	DOCKER = docker
endif

build:
	${DOCKER} build -f=Dockerfile -t=${IMAGE} ${BUILD_ARGUMENTS} ${CONTEXT}

shell: build
	${DOCKER} run ${RUN_ARGUMENTS} -it ${IMAGE} /bin/bash

test: build
	${DOCKER} run ${RUN_ARGUMENTS} -it ${IMAGE} py.test tests

run: build
	${DOCKER} run ${RUN_ARGUMENTS} -it ${IMAGE}

push:
	docker push ${IMAGE}
