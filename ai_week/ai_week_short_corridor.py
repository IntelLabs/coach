
import numpy as np
import gym
from gym import spaces


class ShortCorridorEnv(gym.Env):
    #metadata = {'render.modes': ['human']}

    def __init__(self):

        self.REVERSE_STATE = 1
        self.GOAL_STATE = 3
        self.NUM_STATES = self.GOAL_STATE + 1

        self.observation_space = spaces.Box(0, 1, shape=(self.NUM_STATES,))
        self.action_space = spaces.Discrete(2)

        self.current_state = 0
        self.goal_reached = False
        self.max_steps = 500

    def _get_obs(self):
        self.observation = np.zeros((self.NUM_STATES,))
        #self.observation[self.current_state] = 1 # For bug test = 0
        return self.observation

    # def _terminate(self):
    #     return self.steps >= self.max_steps

    def step(self, action):
        step = -1 if action == 0 else 1

        if self.current_state == self.REVERSE_STATE:
            step = - step

        self.current_state += step

        self.current_state = max(0, self.current_state)
        self.current_state = self.current_state % (self.NUM_STATES)

        # For bug test observation = 0
        observation = self._get_obs()
        reward = -1
        done = self.current_state >= self.GOAL_STATE #or self._terminate()
        self.steps += 1
        #print('steps: ', self.steps)
        return observation, reward, done, {}

    def reset(self):
        self.goal_reached = False
        self.current_state = 0
        self.steps = 0
        observation = self._get_obs()
        return observation

    # def render(self, mode='human'):
    #     corridor = ""
    #     for i in range(self.NUM_STATES):
    #         marker = " "
    #         if self.current_state == i:
    #             marker = "x"
    #         if i == self.REVERSE_STATE:
    #             corridor += "{" + marker + "}"
    #         else:
    #             corridor += "[" + marker + "]"
    #
    #     print(corridor)


