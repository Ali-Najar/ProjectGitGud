from envs.move_env import MoveEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv,VecNormalize



move_env = MoveEnv(speed=1)

move_env = DummyVecEnv([lambda: MoveEnv()])

move_env = VecNormalize.load("vec_normalize_stats_move_v6.pkl", move_env)


model = PPO.load("ppo_dark_souls_move_v6", env=move_env)

obs = move_env.reset()

while True:  # Run for a certain number of steps
    action, _ = model.predict(obs)  # Predict the next action
    obs, rewards, done, info = move_env.step(action)  # Take the action in the environment

    if done:
        obs = move_env.reset()
        break

move_env.close()