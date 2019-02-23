# Container Images

In this directory we've put together several different Dockerfile's that can be used to build
containers that have coach and other environments/dependencies installed.  How to build these
and what each contains is defined below:

## default `Dockerfile`
* `make build` to create the image
* will create a basic Coach installation along with Gym (atari), Mujoco, and Vizdoom environments.
* useful for running unit/integration tests  `make unit_tests` to run these in the container
* `make shell` will launch this container locally, and provide a bash shell prompt.
* includes GPU support (derives from `Dockerfile.base` which is a CUDA ubuntu 16.04 derived image)

## `Dockerfile.mujoco_environment`
* `docker build --build-arg MUJOCO_KEY=${MUJOCO_KEY} -f docker/Dockerfile.mujoco_environment .`
  from the parent dir to create the image
* contains mujoco environment and Coach.
* you need to supply your own license key (base64 encrypted) as an environment variable `MUJOCO_KEY`
  to ensure you get the complete Mujoco environment

## `Dockerfile.gym_environment`
* `docker build -f docker/Dockerfile.gym_environment .` from the parent dir to create the image
* contains OpenAI Gym environment (and all extras) and Coach.

## `Dockerfile.doom_environment`
* `docker build -f docker/Dockerfile.doom_environment .` from the parent dir to create the image
* contains vizdoom environment and Coach.

## `Dockerfile.starcraft_environment`
* `docker build -f docker/Dockerfile.starcraft_environment .` from the parent dir to create the image
* contains StarcraftII environment and Coach.

## `Dockerfile.carla_environment`
* `docker build -f docker/Dockerfile.carla_environment .` from the parent dir to create the image
* contains CARLA driving simulator environment and Coach.
