import numpy as np

def combination_lock(env, correct_prob):
    """
    Heuristic to solve a combination lock environment by selecting the next number in the combination 
    with a given probability, or a random number otherwise.

    Parameters
    ----------
    env : gym.Env
        The environment instance of the combination lock.
    correct_prob : float
        The probability of selecting the correct next number in the combination. Must be between 0 and 1.

    Returns
    -------
    action : int
        The selected action, either the correct next number in the combination or a randomly chosen number 
        from the action space.
    """
    next_number = env.unwrapped.combination[env.combo_step]
    if np.random.random(1) <= correct_prob:
        action = next_number
    else:
        action = env.action_space.sample()
        while action == next_number:
            action = env.action_space.sample()
    return action