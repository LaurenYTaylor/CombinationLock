from gymnasium.envs.registration import register

register(
     id="CombinationLock-v0",
     entry_point="combination_lock:CombinationLockEnv"
)
