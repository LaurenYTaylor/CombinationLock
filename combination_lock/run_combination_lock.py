import gymnasium as gym
import numpy as np
from guide_heuristic import combination_lock
from gymnasium.envs.registration import register
from stable_baselines3 import PPO

# Train a PPO agent or use an oracle to solve the combination lock environment

register(
     id="CombinationLock-v0",
     entry_point="combination_lock:CombinationLockEnv"
)

env = gym.make("CombinationLock-v0", horizon=10)

method = "rl"

if method == "rl":
    model = PPO("MlpPolicy", env, verbose=1, device="cpu", batch_size=256)
    model.learn(total_timesteps=500000)
else:
    model = combination_lock
    correct_prob = 0.5

def main():
    num_eps = 10000
    rewards = []
    for ep in range(num_eps):
        obs, _ = env.reset()
        done = False
        ep_reward = 0
        while not done:
            if method == "rl":
                action = model.predict(obs, deterministic=True)[0]
            else:
                action = model(env, correct_prob)
            next_obs, reward, term, _, _ = env.step(action)
            done = term
            obs = next_obs
            ep_reward += reward
        rewards.append(ep_reward)
    print(f"Mean Reward: {np.mean(rewards)}")
if __name__ == "__main__":
    main()