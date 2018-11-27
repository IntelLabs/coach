Adding a New Environment
========================

Adding a new environment to Coach is as easy as solving CartPole.

There are essentially two ways to integrate new environments to Coach:

Using the OpenAI Gym API
------------------------

If your environment is already using the OpenAI Gym API, you are already good to go.
When selecting the environment parameters in the preset, use :code:`GymEnvironmentParameters()`,
and pass the path to your environment source code using the level parameter.
You can specify additional parameters for your environment using the additional_simulator_parameters parameter.
Take for example the definition used in the :code:`Pendulum_HAC` preset:

.. code-block:: python

        env_params = GymEnvironmentParameters()
        env_params.level = "rl_coach.environments.mujoco.pendulum_with_goals:PendulumWithGoals"
        env_params.additional_simulator_parameters = {"time_limit": 1000}

Using the Coach API
-------------------

There are a few simple steps to follow, and we will walk through them one by one.
As an alternative, we highly recommend following the corresponding
`tutorial <https://github.com/NervanaSystems/coach/blob/master/tutorials/2.%20Adding%20an%20Environment.ipynb>`_
in the GitHub repo.

1. Create a new class for your environment, and inherit the Environment class.

2. Coach defines a simple API for implementing a new environment, which are defined in environment/environment.py.
   There are several functions to implement, but only some of them are mandatory.

   Here are the important ones:

   .. code-block:: python

            def _take_action(self, action_idx: ActionType) -> None:
                """
                An environment dependent function that sends an action to the simulator.
                :param action_idx: the action to perform on the environment
                :return: None
                """

            def _update_state(self) -> None:
                """
                Updates the state from the environment.
                Should update self.observation, self.reward, self.done, self.measurements and self.info
                :return: None
                """

            def _restart_environment_episode(self, force_environment_reset=False) -> None:
                """
                Restarts the simulator episode
                :param force_environment_reset: Force the environment to reset even if the episode is not done yet.
                :return: None
                """

            def _render(self) -> None:
                """
                Renders the environment using the native simulator renderer
                :return: None
                """

            def get_rendered_image(self) -> np.ndarray:
                """
                Return a numpy array containing the image that will be rendered to the screen.
                This can be different from the observation. For example, mujoco's observation is a measurements vector.
                :return: numpy array containing the image that will be rendered to the screen
                """

3. Create a new parameters class for your environment, which inherits the EnvironmentParameters class.
   In the __init__ of your class, define all the parameters you used in your Environment class.
   Additionally, fill the path property of the class with the path to your Environment class.
   For example, take a look at the EnvironmentParameters class used for Doom:

    .. code-block:: python

            class DoomEnvironmentParameters(EnvironmentParameters):
            def __init__(self):
                super().__init__()
                self.default_input_filter = DoomInputFilter
                self.default_output_filter = DoomOutputFilter
                self.cameras = [DoomEnvironment.CameraTypes.OBSERVATION]

            @property
            def path(self):
                return 'rl_coach.environments.doom_environment:DoomEnvironment'
    

4.  And that's it, you're done. Now just add a new preset with your newly created environment, and start training an agent on top of it.
