from envs.cam_env import CamEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv,VecNormalize



cam_env = CamEnv(speed=1)

cam_env = DummyVecEnv([lambda: CamEnv()])

cam_env = VecNormalize.load("vec_normalize_stats_cam_isolated_v2.pkl", cam_env)


model = PPO.load("ppo_dark_souls_cam_isolated_v2", env=cam_env)

obs = cam_env.reset()

while True:  # Run for a certain number of steps
    action, _ = model.predict(obs)  # Predict the next action
    obs, rewards, done, info = cam_env.step(action)  # Take the action in the environment
    if done:
        obs = cam_env.reset()
        break

cam_env.close()