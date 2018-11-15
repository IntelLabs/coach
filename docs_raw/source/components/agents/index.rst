Agents
======

Coach supports many state-of-the-art reinforcement learning algorithms, which are separated into three main classes -
value optimization, policy optimization and imitation learning.
A detailed description of those algorithms can be found by navigating to each of the algorithm pages.

.. image:: /_static/img/algorithms.png
   :width: 600px
   :align: center

.. toctree::
   :maxdepth: 1
   :caption: Agents

   policy_optimization/ac
   imitation/bc
   value_optimization/bs_dqn
   value_optimization/categorical_dqn
   imitation/cil
   policy_optimization/cppo
   policy_optimization/ddpg
   other/dfp
   value_optimization/double_dqn
   value_optimization/dqn
   value_optimization/dueling_dqn
   value_optimization/mmc
   value_optimization/n_step
   value_optimization/naf
   value_optimization/nec
   value_optimization/pal
   policy_optimization/pg
   policy_optimization/ppo
   value_optimization/rainbow
   value_optimization/qr_dqn


.. autoclass:: rl_coach.base_parameters.AgentParameters

.. autoclass:: rl_coach.agents.agent.Agent
   :members:
   :inherited-members:

