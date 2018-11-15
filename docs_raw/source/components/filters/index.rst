Filters
=======

.. toctree::
   :maxdepth: 1
   :caption: Filters

   input_filters
   output_filters

Filters are a mechanism in Coach that allows doing pre-processing and post-processing of the internal agent information.
There are two filter categories -

* **Input filters** - these are filters that process the information passed **into** the agent from the environment.
  This information includes the observation and the reward. Input filters therefore allow rescaling observations,
  normalizing rewards, stack observations, etc.

* **Output filters** - these are filters that process the information going **out** of the agent into the environment.
  This information includes the action the agent chooses to take. Output filters therefore allow conversion of
  actions from one space into another. For example, the agent can take :math:`N` discrete actions, that will be mapped by
  the output filter onto :math:`N` continuous actions.

Filters can be stacked on top of each other in order to build complex processing flows of the inputs or outputs.

.. image:: /_static/img/filters.png
   :width: 350px
   :align: center

