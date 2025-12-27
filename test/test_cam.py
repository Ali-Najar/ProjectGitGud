from envs.cam_env import CamEnv
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv,VecNormalize
import pydirectinput as pdir
import os 

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

pdir.PAUSE = 0.0

# --- CONFIGURATION ---
AGENT_TYPE = "camera"
MODEL_NAME = "dqn_cam_v1"
BASE_DIR = f"model_weights/{AGENT_TYPE}"

MODEL_PATH = f"{BASE_DIR}/{MODEL_NAME}.zip"
STATS_PATH = f"{BASE_DIR}/{MODEL_NAME}_vecnorm.pkl"

cam_env = CamEnv(speed=1)

cam_env = DummyVecEnv([lambda: cam_env])

if os.path.exists(STATS_PATH):
    print(f"Loading normalization stats from {STATS_PATH}...")
    cam_env = VecNormalize.load(STATS_PATH, cam_env)
    cam_env.training = False   
    cam_env.norm_reward = False # We want to see raw rewards, not normalized ones
else:
    print("Warning: No normalization stats found! Agent performance will be poor.")

if os.path.exists(MODEL_PATH):
    print(f"Loading DQN model from {MODEL_PATH}...")
    model = DQN.load(MODEL_PATH, env=cam_env)
else:
    print(f"Error: Model file not found at {MODEL_PATH}")
    exit()

obs = cam_env.reset()


try:
    while True:  # Run for a certain number of steps
        # random_counter = 128
        # while random_counter > 0:
        #     action = [cam_env.action_space.sample()]
            
        #     # Update your counter so the loop eventually ends
        #     random_counter -= 1
        #     obs, rewards, done, info = cam_env.step(action)  # Take the action in the environment
        #     if done:
        #         obs = cam_env.reset()
        # random_counter = 128
        # while random_counter > 0:
        action, _ = model.predict(obs, deterministic=True)  # Predict the next action
        # random_counter -= 1
        obs, rewards, done, info = cam_env.step(action)  # Take the action in the environment
        if done:
            obs = cam_env.reset()
            
except KeyboardInterrupt:
    print("Stopped.")
    cam_env.close()