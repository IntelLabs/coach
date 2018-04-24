Adding a new environment to Coach is as easy as solving CartPole. 

There are a few simple steps to follow, and we will walk through them one by one.

1.  Coach defines a simple API for implementing a new environment which is defined in environment/environment_wrapper.py.
    There are several functions to implement, but only some of them are mandatory. 

    Here are the important ones:

            def _take_action(self, action_idx):
                """
                An environment dependent function that sends an action to the simulator.
                :param action_idx: the action to perform on the environment.
                :return: None
                """
                pass

            def _preprocess_observation(self, observation):
                """
                Do initial observation preprocessing such as cropping, rgb2gray, rescale etc.
                Implementing this function is optional.
                :param observation: a raw observation from the environment
                :return: the preprocessed observation
                """
                return observation

            def _update_state(self):
                """
                Updates the state from the environment.
                Should update self.observation, self.reward, self.done, self.measurements and self.info
                :return: None
                """
                pass

            def _restart_environment_episode(self, force_environment_reset=False):
                """
                :param force_environment_reset: Force the environment to reset even if the episode is not done yet.
                :return:
                """
                pass

            def get_rendered_image(self):
                """
                Return a numpy array containing the image that will be rendered to the screen.
                This can be different from the observation. For example, mujoco's observation is a measurements vector.
                :return: numpy array containing the image that will be rendered to the screen
                """
                return self.observation


2.  Make sure to import the environment in environments/\_\_init\_\_.py:
        
        from doom_environment_wrapper import *
        
    Also, a new entry should be added to the EnvTypes enum mapping the environment name to the wrapper's class name:
        
        Doom = "DoomEnvironmentWrapper"
    
                
3. In addition a new configuration class should be implemented for defining the environment's parameters and placed in configurations.py. 
For instance, the following is used for Doom:

        class Doom(EnvironmentParameters):
            type = 'Doom'
            frame_skip = 4
            observation_stack_size = 3
            desired_observation_height = 60
            desired_observation_width = 76
            
4. And that's it, you're done. Now just add a new preset with your newly created environment, and start training an agent on top of it. 
