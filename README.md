# CombinationLock
A toy combination lock environment.

The environment can be installed by:
```
pip install -i https://test.pypi.org/simple/ combination-lock
```
You can create the environment with:
```
register(
     id="CombinationLock-v0",
     entry_point="combination_lock:CombinationLockEnv"
)

env = gym.make("CombinationLock-v0", horizon=10)
```

An example of an RL agent or heuristic solving the environment can be found in run_combination_lock.py.
