# Scaling out rollout workers

This document contains some options for how we could implement horizontal scaling of rollout workers in coach, though most details are not specific to coach. A few options are laid out, my current suggestion would be to start with Option 1, and move on to Option 1a or Option 1b as required.

## Off Policy Algorithms

### Option 1 - master polls file system

- one master process samples memories and updates the policy
- many worker processes execute rollouts
- coordinate using a single shared networked file system: nfs, ceph, dat, s3fs, etc.
- policy sync communication method:
  - master process occasionally writes policy to shared file system
  - worker processes occasionally read policy from shared file system
  - prevent workers from reading a policy which has not been completely written to disk using either:
    - redis lock
    - write to temporary files and then rename
- rollout memories:
  - sync communication method:
    - worker processes write rollout memories as they are generated to shared filesystem
    - master process occasionally reads rollout memories from shared file system
    - master process must be resilient to corrupted or incompletely written memories
  - sampling method:
    - master process keeps all rollouts in memory utilizing existing coach memory classes
- control flow:
  - master:
    - run training updates interleaved with loading of any newly available rollouts in memory
    - periodically write policy to disk
  - workers:
    - periodically read policy from disk
    - evaluate rollouts and write them to disk
- ops:
  - kubernetes yaml, kml, docker compose, etc
  - a default shared file system can be provided, while allowing the user to specify something else if desired
  - a default method of launching the workers and master (in kubernetes, gce, aws, etc) can be provided

#### Pros

- very simple to implement, infrastructure already available in ai-lab-kubernetes
- fast enough for proof of concept and iteration of interface design
- rollout memories are durable and can be easily reused in later off policy training
- if designed properly, there is a clear path towards:
  - decreasing latency using in-memory store (option 1a/b)
  - increasing rollout memory size using distributed sampling methods (option 1c)

#### Cons

- file system interface incurs additional latency. rollout memories must be written to disk, and later read from disk, instead of going directly from memory to memory.
- will require modifying standard control flow. there will be an impact on algorithms which expect particular training regimens. Specifically, algorithms which are sensitive to the number of update steps between target/online network updates
- will not be particularly efficient in strictly on policy algorithms where each rollout must use the most recent policy available

### Option 1a - master polls (redis) list

- instead of using a file system as in Option 1, redis lists can be used
- policy is stored as a single key/value pair (locking no longer necessary)
- rollout memory communication:
  - workers: redis list push
  - master: redis list len, redis list range
- note: many databases are interchangeable with redis protocol: google memorystore, aws elasticache, etc.
- note: many databases can implement this interface with minimal glue: SQL, any objectstore, etc.

#### Pros

- lower latency than disk since it is all in memory
- clear path toward scaling to large number of workers
- no concern about reading partially written rollouts
- no synchronization or additional threads necessary, though an additional thread would be helpful for concurrent reads from redis and training
- will be slightly more efficient in the case of strictly on policy algorithms

#### Cons

- more complex to set up, especially if you are concerned about rollout memory durability

### Option 1b - master subscribes to (redis) pub sub

- instead of using a file system as in Option 1, redis pub sub can be used
- policy is stored as a single key/value pair (locking no longer necessary)
- rollout memory communication:
  - workers: redis publish
  - master: redis subscribe
- no synchronization necessary, however an additional thread would be necessary?
  - it looks like the python client might handle this already, would need further investigation
- note: many possible pub sub systems could be used with different characteristics under specific contexts: kafka, google pub/sub, aws kinesis, etc

#### Pros

- lower latency than disk since it is all in memory
- clear path toward scaling to large number of workers
- no concern about reading partially written rollouts
- will be slightly more efficient in the case of strictly on policy algorithms

#### Cons

- more complex to set up then shared file system
- on its own, does not persist worker rollouts for future off policy training

### Option 1c - distributed rollout memory sampling

- if rollout memories do not fit in memory of a single machine, a distributed storage and sampling method would be necessary
- for example:
  - rollout memory store: redis set add
  - rollout memory sample: redis set randmember

#### Pros

- capable of taking advantage of rollout memory larger than the available memory of a single machine
- reduce resource constraints on training machine

#### Cons

- distributed versions of each memory type/sampling method need to be custom built
- off-the-shelf implementations may not be available for complex memory types/sampling methods

### Option 2 - master listens to workers

- rollout memories:
  - workers send memories directly to master via: mpi, 0mq, etc
  - master policy thread listens for new memories and stores them in shared memory
- policy sync communication memory:
  - master policy occasionally sends policies directly to workers via: mpi, 0mq, etc
  - master and workers must synchronize so that all workers are listening when the master is ready to send a new policy

#### Pros

- lower latency than option 1 (for a small number of workers)
- will potentially be the optimal choice in the case of strictly on policy algorithms with relatively small number of worker nodes (small enough that more complex communication typologies would be necessary: rings, p2p, etc)

#### Cons

- much less robust and more difficult to debug requiring lots of synchronization
- much more difficult to be resiliency worker failure
- more custom communication/synchronization code
- as the number of workers scale up, a larger and larger fraction of time will be spent waiting and synchronizing

### Option 3 - Ray

#### Pros

- Ray would allow us to easily convert our current algorithms to distributed versions, with minimal change to our code.

#### Cons

- performance from naïve/simple use would be very similar to Option 2
- nontrivial to replace with a higher performance system if desired. Additional performance will require significant code changes.

## On Policy Algorithms

TODO
