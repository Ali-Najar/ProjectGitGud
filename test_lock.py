from envs.lock_env import LockEnv
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv,VecNormalize
import pydirectinput as pdir
import os

pdir.PAUSE = 0.0

# --- CONFIGURATION ---
AGENT_TYPE = "locking"
MODEL_NAME = "dqn_lock_v1"
BASE_DIR = f"model_weights/{AGENT_TYPE}"

MODEL_PATH = f"{BASE_DIR}/{MODEL_NAME}.zip"
STATS_PATH = f"{BASE_DIR}/{MODEL_NAME}_vecnorm.pkl"

lock_env = LockEnv(speed=1)

lock_env = DummyVecEnv([lambda: lock_env])

if os.path.exists(STATS_PATH):
    print(f"Loading normalization stats from {STATS_PATH}...")
    lock_env = VecNormalize.load(STATS_PATH, lock_env)
    lock_env.training = False   
    lock_env.norm_reward = False # We want to see raw rewards, not normalized ones
else:
    print("Warning: No normalization stats found! Agent performance will be poor.")

if os.path.exists(MODEL_PATH):
    print(f"Loading DQN model from {MODEL_PATH}...")
    model = DQN.load(MODEL_PATH, env=lock_env)
else:
    print(f"Error: Model file not found at {MODEL_PATH}")
    exit()

obs = lock_env.reset()


try:
    while True:  # Run for a certain number of steps
        random_counter = 128
        while random_counter > 0:
            action = [lock_env.action_space.sample()]
            
            # Update your counter so the loop eventually ends
            random_counter -= 1
            obs, rewards, done, info = lock_env.step(action)  # Take the action in the environment
            if done:
                obs = lock_env.reset()
        random_counter = 128
        # while random_counter > 0:
        #     action, _ = model.predict(obs, deterministic=True)  # Predict the next action
        #     random_counter -= 1
        #     obs, rewards, done, info = lock_env.step(action)  # Take the action in the environment
        #     if done:
        #         obs = lock_env.reset()
except KeyboardInterrupt:
    print("Stopped.")
    lock_env.close()