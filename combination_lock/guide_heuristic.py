import numpy as np

def combination_lock(env, _):
    next_number = env.unwrapped.combination[env.combo_step]
    if np.random.random(1) <= (.75)**(1/3):
        action = next_number
    else:
        action = env.action_space.sample()
        while action == next_number:
            action = env.action_space.sample()
    return action