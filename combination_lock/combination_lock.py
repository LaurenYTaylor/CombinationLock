import gymnasium as gym
import numpy as np
from gymnasium import spaces

#####################################################################################################
# Docstrings were largely generated using ChatGPT (GPT3.5).                                         #
# OpenAI. (2024). Docstrings for CombinationLock environment. Retrieved from ChatGPT on 22 May 2024.#
#####################################################################################################

class CombinationLockEnv(gym.Env):
    """
    A gymnasium environment representing a combination lock.
    Attributes
        ----------
        horizon : int
            The length of the combination lock.
        observation_space : gym.spaces.Box
            The observation space representing the numbers chosen so far. Initialized to a vector of -1s of length `horizon`.
        action_space : gym.spaces.Box
            The action space representing the possible choices for the next number in the combination. Each action is a 
            number between 0 and 9.
        render_mode : str or None
            The mode in which to render the environment. Must be one of the values in `self.metadata["render_modes"]` 
            or None.
        combination : numpy.ndarray
            The correct combination to unlock the lock, represented as a sequence of numbers from 0 to `horizon`-1.
    """
    metadata = {"render_modes": []}

    def __init__(self, render_mode=None, horizon=5):
        """
        Initialize the combination lock environment.

        Parameters
        ----------
        render_mode : str or None, optional
            The mode in which to render the environment. Must be one of the values in `self.metadata["render_modes"]` 
            or None. Defaults to None.
        horizon : int, optional
            The length of the combination lock, i.e., the number of steps required to unlock it. Defaults to 5.
        """
        self.horizon = horizon  # The length of the combination lock

        # Observation is the numbers chosen thus far, initialised to [-1]*self.horizon
        self.observation_space = spaces.Box(low=-1, high=9, shape=(self.horizon,))

        # 10 actions indicating which number to choose next out of 0-9
        self.action_space = spaces.Box(low=0, high=1, shape=(self.horizon,))

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # Combination is always 0,1,2,...,n-1
        self.combination = np.array(range(self.horizon))

    def reset(self, seed=None, options=None):
        """
        Reset the environment to its initial state.

        Parameters
        ----------
        seed : int, optional
            The seed for the environment's random number generator. Defaults to None.
        options : dict, optional
            Additional options for resetting the environment. Defaults to None.

        Returns
        -------
        observation : numpy.ndarray
            The initial observation of the environment, represented as a vector of -1s of length `horizon`.
        info : dict
            An empty dictionary, provided for compatibility with the Gym API.
        """
        super().reset(seed=seed)
        observation = np.array([-1]*self.horizon)
        self.combo_step = 0
        return observation, {}
    
    def step(self, action):
        """
        Take an action in the environment and return the resulting observation, reward, and termination status.

        Parameters
        ----------
        action : int or numpy.ndarray
            The action to take. If it is an array, the function will use the index of the maximum value as the action.

        Returns
        -------
        observation : numpy.ndarray
            The current state of the environment after taking the action, represented as a vector of length `horizon`.
        reward : int
            The reward obtained from taking the action. It is 1 if the combination is correctly completed, otherwise 0.
        terminated : bool
            Whether the episode has terminated (either by correctly completing the combination or making a wrong move).
        truncated : bool
            Always False, provided for compatibility with the Gym API.
        info : dict
            An empty dictionary, provided for compatibility with the Gym API.
        """
        try:
            if len(action) > 0:
                action = np.argmax(action)
        except TypeError:
            action = action
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

    