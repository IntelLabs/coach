Network Design
==============

Each agent has at least one neural network, used as the function approximator, for choosing the actions.
The network is designed in a modular way to allow reusability in different agents.
It is separated into three main parts:

* **Input Embedders** - This is the first stage of the network, meant to convert the input into a feature vector representation.
  It is possible to combine several instances of any of the supported embedders, in order to allow varied combinations of inputs.

    There are two main types of input embedders: 

    1. Image embedder - Convolutional neural network. 
    2. Vector embedder - Multi-layer perceptron. 


* **Middlewares** - The middleware gets the output of the input embedder, and processes it into a different representation domain,
  before sending it through the output head. The goal of the middleware is to enable processing the combined outputs of
  several input embedders, and pass them through some extra processing.
  This, for instance, might include an LSTM or just a plain simple FC layer.

* **Output Heads** - The output head is used in order to predict the values required from the network.
  These might include action-values, state-values or a policy. As with the input embedders,
  it is possible to use several output heads in the same network. For example, the *Actor Critic* agent combines two
  heads - a policy head and a state-value head.
  In addition, the output heads defines the loss function according to the head type.

  ​
.. image:: /_static/img/network.png
   :width: 400px
   :align: center

Keeping Network Copies in Sync
------------------------------

Most of the reinforcement learning agents include more than one copy of the neural network.
These copies serve as counterparts of the main network which are updated in different rates,
and are often synchronized either locally or between parallel workers. For easier synchronization of those copies,
a wrapper around these copies exposes a simplified API, which allows hiding these complexities from the agent.
In this wrapper, 3 types of networks can be defined:

* **online network** - A mandatory network which is the main network the agent will use

* **global network** - An optional network which is shared between workers in single-node multi-process distributed learning.
  It is updated by all the workers directly, and holds the most up-to-date weights.

* **target network** - An optional network which is local for each worker. It can be used in order to keep a copy of
  the weights stable for a long period of time. This is used in different agents, like DQN for example, in order to
  have stable targets for the online network while training it.


.. image:: /_static/img/distributed.png
   :width: 600px
   :align: center


