import numpy as np
import gymnasium as gym
import actions.cam_actions as cam_actions
import envs.lock_env as lock_env
import address
import pyMeow as pm
import pydirectinput as pdir
import time
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv,VecNormalize
import random
import csv 

pdir.PAUSE = 0.0

process = address.process
AGENT_TYPE = "locking"  # e.g., 'locking', 'walking', 'acting'
MODEL_NAME = "dqn_lock_v1"

pm.w_int(process , address.Address.no_dead, 1)

BASE_DIR = f"model_weights/{AGENT_TYPE}"
MODEL_PATH = f"{BASE_DIR}/{MODEL_NAME}"
STATS_PATH = f"{BASE_DIR}/{MODEL_NAME}_vecnorm.pkl"
LOG_PATH   = f"{BASE_DIR}/training_log.csv"
BUFFER_PATH = f"{BASE_DIR}/{MODEL_NAME}_replay_buffer.pkl"

os.makedirs(BASE_DIR, exist_ok=True)

class RewardCallback(BaseCallback):
    def __init__(self, env, log_path, verbose=0):
        super(RewardCallback, self).__init__(verbose)
        self.rewards = []
        self.normalized_rewards = []
        self.total_steps = 0
        self.env = env
        self.log_path = log_path
        self.end_flag = False
        # Internal tracking for the CURRENT episode
        self.current_episode_reward = 0.0
        self.episode_counter = 0
        self.total_steps = 0

        # Check if the CSV exists to resume
        if os.path.exists(self.log_path):
            with open(self.log_path, mode='r', newline='') as file:
                reader = csv.reader(file)
                data = list(reader)
                
                # Check if file has data (header + at least 1 row)
                if len(data) > 1:
                    last_row = data[-1]
                    try:
                        # Index 0 = Episode, Index 1 = Steps, Index 2 = Reward
                        self.episode_counter = int(last_row[0])
                        self.total_steps = int(last_row[1])
                        print(f"Resuming logging from Episode {self.episode_counter}, Step {self.total_steps}")
                    except ValueError:
                        print("Warning: CSV corrupted. Starting from 0.")
        else:
            # Create new file with headers if it doesn't exist
            with open(self.log_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Episode', 'Steps', 'Total_Reward'])

    def _on_step(self) -> bool:
        # Extract the reward from the environment
        reward = self.env.get_original_reward()[0]
        self.current_episode_reward += reward
        self.total_steps += 1

        done_array = self.locals.get("dones")
        done = done_array[0]

        if done:
            self.episode_counter += 1

            # Save to CSV
            with open(self.log_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([self.episode_counter, self.total_steps, self.current_episode_reward])
            
            print(f"Episode {self.episode_counter} Finished! Return: {self.current_episode_reward:.2f}")
            
            # Reset tracker for next episode
            self.current_episode_reward = 0.0
            
        return True

class LockEnvTrain(lock_env.LockEnv):
    def __init__(self , boss_name="iudex", speed = 1):
        super(LockEnvTrain, self).__init__(boss_name="iudex", speed = 1)
        self.loc = [141.87632751464844,-68.88593292236328,593.8687133789062]
        self.max_steps = 64
        self.current_step = 0
        self.game_speed = speed
        pm.w_float(process, address.Address.global_speed, self.game_speed)
    
    def random_point_2d(self,x0, z0, max_distance):
        # Generate a random distance
        distance = random.uniform(0, max_distance)

        # Generate a random angle
        angle = random.uniform(0, 2 * np.pi)

        # Convert polar coordinates to Cartesian coordinates
        x = x0 + distance * np.cos(angle)
        z = z0 + distance * np.sin(angle)

        if self.state[1] == 1:
            pdir.keyDown(cam_actions.Lock_On[0][0])
            time.sleep(0.02)
            pdir.keyUp(cam_actions.Lock_On[0][0])
        else:
            pass

        return (x,z)

    def reset(self , seed=None , options=None):
        super().reset(seed=seed)
        self.current_step = 0

        # 1. Teleport Player
        x, z = self.random_point_2d(self.loc[0], self.loc[2], 10)
        pm.w_float(process, address.Address.X, x)
        pm.w_float(process, address.Address.Z, z)
        pm.w_float(process, address.Address.Y, self.loc[1])
        
        # 2. Heal for continuous training
        pm.w_int(process, address.Address.hp, pm.r_int(process, address.Address.max_hp))
        
        # 3. Get fresh state
        self.state = self.read_from_memory()
        return self.state, {}
    
    def step(self , action):
        reward = 0
        done = False
        info = {}
        pm.w_int(process , address.Address.hp , pm.r_int(process , address.Address.max_hp))           

        if self.current_step == 0:
            pm.w_float(process, address.Address.global_speed, self.game_speed)

        prev_lock = self.state[1]
        angle = self.state[0]

        if action == 0:
            pdir.keyDown(cam_actions.Lock_On[action][0])
            time.sleep(0.1 / self.game_speed)
            pdir.keyUp(cam_actions.Lock_On[action][0])
        else:
            time.sleep(0.1 / self.game_speed)


        next_state = self.read_from_memory()

        current_lock = next_state[1]

        # if current_lock == 0 and prev_lock == 1:
        #     reward -= 1
        # if current_lock == 1 and prev_lock == 0:
        #     reward += 1
        # if angle > 0.7 and action == 0:
        #     reward -= 0.5
        # if action == 0 and prev_lock == 0 and current_lock == 0:
        #     reward -= 0.5
        # if current_lock == 1 and prev_lock == 1:
        #     reward += 0.1

        # if current_lock == 0 and prev_lock == 1:
        #     reward -= 1
        # elif current_lock == 0:
        #     reward -= 0.1
        # if current_lock == 1 and prev_lock == 0:
        #     reward += 1
        # elif current_lock == 1:
        #     reward += 0.1

        if current_lock == 0:
            reward -= 1
        if current_lock == 1:
            reward += 1

        if pm.r_int(process, address.Address.hp)<=0:
            done = True

        truncated = False
        self.current_step += 1
        if self.current_step >= self.max_steps:
            truncated = True

        self.state = next_state

        return next_state, reward, done, truncated , info

env = LockEnvTrain(speed=1)
env = DummyVecEnv([lambda: env])

# normalized_env = VecNormalize(env, norm_obs=True, norm_reward=True, gamma=0.95)

checkpoint_callback = CheckpointCallback(
    save_freq=1000, 
    save_path=BASE_DIR, 
    name_prefix=MODEL_NAME
)

if os.path.exists(STATS_PATH):
    print(f"Loading normalization stats from {STATS_PATH}...")
    env = VecNormalize.load(STATS_PATH, env)
    env.training = True # Ensure we are updating stats during training
    env.norm_reward = False
else:
    print("Creating new normalization stats...")
    env = VecNormalize(env, norm_obs=True, norm_reward=False, gamma=0.99)

# 2. Load/Create Model
zip_path = MODEL_PATH + ".zip"
if os.path.exists(zip_path):
    print(f"Loading existing DQN model from {zip_path}...")
    model = DQN.load(zip_path, env=env,
                      custom_objects={
                        "exploration_rate": 0.01,
                        "exploration_initial_eps": 0.01,
                        "exploration_final_eps": 0.01
                            })
    
    # 3. Load Replay Buffer (Only if model exists)
    if os.path.exists(BUFFER_PATH):
        print(f"Loading replay buffer from {BUFFER_PATH}...")
        try:
            model.load_replay_buffer(BUFFER_PATH)
            print(f"Buffer loaded with {model.replay_buffer.size()} transitions.")
        except Exception as e:
            print(f"Warning: Could not load replay buffer: {e}")
            print("Starting with empty buffer.")
else:
    print(f"Creating new DQN agent...")
    model = DQN(
        "MlpPolicy", 
        env, 
        learning_rate=3e-4, 
        buffer_size=100000,
        learning_starts=1000,
        batch_size=64,
        target_update_interval=1024,
        exploration_fraction=0.5, 
        verbose=1
    )

custom_callback = RewardCallback(env=env, log_path=LOG_PATH, verbose=1)

callbacks = [custom_callback, checkpoint_callback]


print("Starting Training...")
try:
    # 50k steps is a good start. 4096 is too low for DQN.
    model.learn(total_timesteps=2**12, callback=callbacks, reset_num_timesteps=False)
except KeyboardInterrupt:
    print("Training interrupted manually.")

# 4. Save Everything
print(f"Saving model to {MODEL_PATH}...")
model.save(MODEL_PATH)

print(f"Saving stats to {STATS_PATH}...")
env.save(STATS_PATH)

print(f"Saving replay buffer to {BUFFER_PATH}...")
model.save_replay_buffer(BUFFER_PATH)

print("Done.")
# ent_coef = 0.02

# model = PPO.load("ppo_dark_souls_lock_isolated_v1", env=env)

# params = model.get_parameters()

# model = PPO('MlpPolicy',learning_rate=0.001,n_steps=512 , batch_size=64 ,  n_epochs=10 , ent_coef=ent_coef ,env = env, verbose=1)

# model.set_parameters(params)


# model.learn(total_timesteps=4096 , callback=custom_callback)

# env.save("vec_normalize_stats_lock_isolated_v1.pkl")

# model.save("ppo_dark_souls_lock_isolated_v1")