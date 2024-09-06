from envs.lock_env import LockEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv,VecNormalize



lock_env = LockEnv(speed=1)

lock_env = DummyVecEnv([lambda: LockEnv()])

lock_env = VecNormalize.load("vec_normalize_stats_lock_isolated_v1.pkl", lock_env)


model = PPO.load("ppo_dark_souls_lock_isolated_v1", env=lock_env)

obs = lock_env.reset()

while True:  # Run for a certain number of steps
    action, _ = model.predict(obs)  # Predict the next action
    obs, rewards, done, info = lock_env.step(action)  # Take the action in the environment

    if done:
        obs = lock_env.reset()
        break

lock_env.close()