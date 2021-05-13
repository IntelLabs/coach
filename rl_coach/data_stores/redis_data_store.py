#
# Copyright (c) 2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import time
import uuid

import redis

from rl_coach.architectures.tensorflow_components.savers import GlobalVariableSaver
from rl_coach.data_stores.data_store import DataStore, DataStoreParameters


class RedisDataStoreParameters(DataStoreParameters):
    def __init__(
        self,
        ds_params,
        redis_address: str = "",
        redis_port: int = 6379,
        redis_channel: str = "data-store-channel-{}".format(uuid.uuid4()),
    ):
        super().__init__(
            ds_params.store_type,
            ds_params.orchestrator_type,
            ds_params.orchestrator_params,
        )
        self.redis_address = redis_address
        self.redis_port = redis_port
        self.redis_channel = redis_channel


class RedisDataStore(DataStore):
    """
    This DataStore sends policies over redis pubsub and get/set.

    Deployment
    ==========
    It assumes that a redis server is already available. We make this assumption because during
    multinode training at this time, redis is already used for communicating replay memories.

    Communication
    =============

    A redis pubsub channel is used by the training worker to signal to the rollout workers that a
    new policy is ready. When this occurs, a new policy is loaded from the redis key/value store
    where key is the same as the pubsub channel. Originally, just the pubsub was used, but that
    could result in a race condition where the master worker publishes the first policy and waits
    for the rollout workers to submit all rollouts, while a delayed rollout worker waits for the
    first policy since it subscribed to the channel after the initial policy was published.
    """

    def __init__(self, params: RedisDataStoreParameters):
        self.params = params
        self.saver = None
        self._end_of_policies = False

        # NOTE: a connection is not attempted at this stage because the address and port are likely
        # not available yet. This is because of how the kubernetes orchestrator works. At the time
        # of parameter construction, the address and port are not yet known since they are copied
        # out of the redis memory backend after it is deployed. One improvement would be to use
        # two separate redis deployments independently, and let this class deploy its own redis.

    def _connect(self):
        """
        Connect to redis and subscribe to the pubsub channel
        """
        self.redis_connection = redis.Redis(
            self.params.redis_address, self.params.redis_port
        )
        self.pubsub = self.redis_connection.pubsub(ignore_subscribe_messages=True)
        self.pubsub.subscribe(self.params.redis_channel)

        self._end_of_policies = False

    def deploy(self):
        """
        For now, this data store does not handle its own deployment, it piggybacks off of the redis
        memory backend
        """
        return True

    def undeploy(self):
        """
        For now, this data store does not handle its own deployment, it piggybacks off of the redis
        memory backend
        """
        pass

    def save_to_store(self):
        """
        save_to_store and load_from_store are not used in the case where the data stored needs to
        synchronize checkpoints saved to disk into a central file system, and not used here
        """
        pass

    def load_from_store(self):
        """
        save_to_store and load_from_store are not used in the case where the data stored needs to
        synchronize checkpoints saved to disk into a central file system, and not used here
        """
        pass

    def save_policy(self, graph_manager):
        """
        Serialize the policy in graph_manager, set it as the latest policy and publish a new_policy
        event
        """
        if self.saver is None:
            self.saver = GlobalVariableSaver()

            # TODO: only subscribe if this data store is being used to publish policies
            self._connect()
            self.pubsub.unsubscribe(self.params.redis_channel)

        policy_string = self.saver.to_string(graph_manager.sess)
        self.redis_connection.set(self.params.redis_channel, policy_string)
        self.redis_connection.publish(self.params.redis_channel, "new_policy")

    def _load_policy(self, graph_manager) -> bool:
        """
        Get the most recent policy from redis and loaded into the graph_manager
        """
        policy_string = self.redis_connection.get(self.params.redis_channel)
        if policy_string is None:
            return False

        self.saver.from_string(graph_manager.sess, policy_string)
        return True

    def load_policy(self, graph_manager, require_new_policy=True, timeout=0):
        """
        :param graph_manager: the graph_manager to load the policy into
        :param require_new_policy: if True, only load a policy if it hasn't been loaded in this
        process yet before.
        :param timeout: Will only try to load the policy once if timeout is None, otherwise will
        retry for timeout seconds
        """
        if self.saver is None:
            # the GlobalVariableSaver needs to be instantiated after the graph is created. For now,
            # it can be instantiated here, but it might be nicer to have a more explicit
            # on_graph_creation_end callback or similar to put it in
            self.saver = GlobalVariableSaver()
            self._connect()

        if not require_new_policy:
            # try just loading whatever policy is available most recently
            if self._load_policy(graph_manager):
                return

        message = "first"
        timeout_ends = time.time() + timeout
        while time.time() < timeout_ends or message == "first":
            message = self.pubsub.get_message()

            if message and message["type"] == "message":
                if message["data"] == b"end_of_policies":
                    self._end_of_policies = True
                    return
                elif message["data"] == b"new_policy":
                    if self._load_policy(graph_manager):
                        return
                    else:
                        raise ValueError("'new_policy' message was sent, but no policy was found.")

            time.sleep(1.0)

        if require_new_policy:
            raise ValueError(
                "Waited for {timeout} seconds on channel {channel}, but no first policy was received.".format(
                    timeout=timeout, channel=self.params.redis_channel
                )
            )

    def end_of_policies(self) -> bool:
        """
        This is used by the rollout workers to detect a message from the training worker signaling
        that training is complete.
        """
        return self._end_of_policies
