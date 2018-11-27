.. _dist-coach-design:

Distributed Coach - Horizontal Scale-Out
========================================

Coach supports the horizontal scale-out of rollout workers using `--distributed_coach` or `-dc` options. Coach uses
three interfaces for horizontal scale-out, which allows for integration with different technologies and flexibility.
These three interfaces are orchestrator, memory backend and data store.

* **Orchestrator** - The orchestrator interface provides basic interaction points for orchestration, scheduling and
  resource management of training and rollout workers in the distributed coach mode. The interactions points define
  how Coach should deploy, undeploy and monitor the workers spawned by Coach.

* **Memory Backend** - This interface is used as the backing store or stream for the memory abstraction in
  distributed Coach. The implementation of this module is mainly used for communicating experiences (transitions
  and episodes) from the rollout to the training worker.

* **Data Store** - This interface is used as a backing store for the policy checkpoints. It is mainly used to
  synchronizing policy checkpoints from the training to the rollout worker.

.. image:: /_static/img/horizontal-scale-out.png
   :width: 800px
   :align: center

Supported Synchronization Types
-------------------------------

Synchronization type refers to the mechanism by which the policy checkpoints are synchronized from the training to the
rollout worker. For each algorithm, it is specified by using the `DistributedCoachSynchronizationType` as a part of
`agent_params.algorithm.distributed_coach_synchronization_type` in the preset. In distributed Coach, two types of
synchronization modes are supported: `SYNC` and `ASYNC`.

* **SYNC** - In this type, the trainer waits for all the experiences to be gathered from distributed rollout workers
  before training a new policy and the rollout workers wait for a new policy before gathering experiences. It is suitable
  for ON policy algorithms.

* **ASYNC** - In this type, the trainer doesn't wait for any set of experiences to be gathered from distributed
  rollout workers and the rollout workers continously gather experiences loading new policies, whenever they become
  available. It is suitable for OFF policy algorithms.
