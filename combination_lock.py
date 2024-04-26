import gymnasium as gym
import numpy as np
from gymnasium import spaces

class CombinationLockEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, render_mode=None, horizon=5):
        self.horizon = horizon  # The length of the combination lock

        # Observation is the numbers chosen thus far, initialised later to [-1]*self.horizon
        self.observation_space = spaces.MultiDiscrete([9]*self.horizon, start=[-1]*self.horizon)

        # 10 actions indicating which number to choose next (0-9)
        self.action_space = spaces.Discrete(10)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.combination = np.random.randint(10, size=(self.horizon,))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        observation = np.array([-1]*self.horizon)
        self.combo_step = 0
        return observation, {}
    
    def step(self, action):
        observation = [-1]*self.horizon
        observation[:self.combo_step] = self.combination[:self.combo_step]
        reward = 0
        if action == self.combination[self.combo_step]:
            observation[self.combo_step] = action
            if self.combo_step == self.horizon-1:
                terminated = True
                reward = 1
            else:
                terminated = False
        else:
            terminated = True        
        self.combo_step += 1
        return np.array(observation), reward, terminated, False, {}

    