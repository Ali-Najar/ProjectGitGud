import numpy as np
import gymnasium as gym
import actions.move_actions as move_actions
import envs.move_env as move_env
import address
import pyMeow as pm
import pydirectinput as pdir
import time
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv,VecNormalize
import random
import os 
import csv

pdir.PAUSE = 0.0

process = address.process
AGENT_TYPE = "move"  # Changed to camera
MODEL_NAME = "dqn_move_v1"

pm.w_int(process , address.Address.no_dead, 1)
pm.w_int(process , address.Address.no_hit, 1)

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

class MoveEnvTrain(move_env.MoveEnv):
    def __init__(self , boss_name="iudex", speed = 1):
        super(MoveEnvTrain, self).__init__(boss_name="iudex", speed = 1)
        self.loc = [141.87632751464844,-68.88593292236328,593.8687133789062]
        self.max_steps = 128
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

        return (x,z)
    
    def random_hp_and_estus_stamina(self):
        return random.randint(0,1) , random.randint(150,self.maximum_hp) , random.randint(0,self.maximum_stamina)
    
    def reset(self , seed=None , options=None):
        super().reset(seed=seed)
        self.current_step = 0

        # pm.w_float(process, address.Address.global_speed, 0.0)

        # 1. Teleport Player
        x, z = self.random_point_2d(self.loc[0], self.loc[2], 10)
        pm.w_float(process, address.Address.X, x)
        pm.w_float(process, address.Address.Z, z)
        pm.w_float(process, address.Address.Y, self.loc[1])
        
        # 3. Get fresh state
        self.state = self.read_from_memory()
        return self.state, {}
    
    
    def step(self , action):

        reward = 0
        done = False
        info = {}

        for act in move_actions.sub_actions[action]:
            pdir.keyDown(act)
        time.sleep(0.1 / self.speed)
        self.release_key()

        next_state = self.read_from_memory()

        distance = np.linalg.norm(next_state[0:3] - next_state[3:6])

        reward -= distance / 10
    
        if distance < 5 and action in [4, 5]:
            reward += 1

        # # if stamina == 0:
        # #     if action <= 15 and action >= 8:
        # #         reward -= 15
        # if hp == 0 and estus == 1:
        #     reward -= abs(distance-30) - 7
        #     if attacking == 1 and action == 1:
        #         reward += 20
        #     # if attacking == 0 and action == 9:
        #     #     reward += 20
        # else:
        #     reward += (4 - distance)*7
        #     if distance < 5:
        #         if action ==2 or action==3:
        #             reward += 20
        #         if action <= 1:
        #             reward += 4
        #         if action == 4:
        #             reward -= 8
        #     if attacking == 1:
        #         if action >= 2 and action <= 3:
        #             reward += 20
        #     else:
        #         if distance > 5:
        #             if action == 0:
        #                 reward += 25
        #         # if distance > 10:
        #             # if action == 8:
        #             #     reward += 30
        # # if attacking == 1:
        #     # if distance < 3 and action <=13 and action >=10:
        #     #     reward -= 20
        #     # if action == 8:
        #     #     reward -= 10

        truncated = False
        self.current_step += 1
        if self.current_step >= self.max_steps:
            truncated = True

        return next_state, reward, done, truncated , info

env = MoveEnvTrain(speed=1)
env = DummyVecEnv([lambda: env])

checkpoint_callback = CheckpointCallback(
    save_freq=1000, 
    save_path=BASE_DIR, 
    name_prefix=MODEL_NAME
)

# 1. Load/Create VecNormalize
if os.path.exists(STATS_PATH):
    print(f"Loading normalization stats from {STATS_PATH}...")
    env = VecNormalize.load(STATS_PATH, env)
    env.training = True 
    env.norm_reward = False
else:
    print("Creating new normalization stats...")
    env = VecNormalize(env, norm_obs=True, norm_reward=False, gamma=0.99)

# 2. Load/Create Model
zip_path = MODEL_PATH + ".zip"
if os.path.exists(zip_path):
    print(f"Loading existing DQN model from {zip_path}...")
    model = DQN.load(zip_path, env=env)
    
    if os.path.exists(BUFFER_PATH):
        print(f"Loading replay buffer from {BUFFER_PATH}...")
        try:
            model.load_replay_buffer(BUFFER_PATH)
        except Exception as e:
            print(f"Warning: Could not load replay buffer: {e}")
else:
    print(f"Creating new DQN agent...")
    model = DQN(
        "MlpPolicy", 
        env, 
        learning_rate=3e-4, 
        buffer_size=100000,
        learning_starts=1000,
        batch_size=32,
        target_update_interval=1024,
        exploration_fraction=0.6, 
        verbose=1
    )

custom_callback = RewardCallback(env=env, log_path=LOG_PATH, verbose=1)

callbacks = [custom_callback, checkpoint_callback]

print("Starting Training...")
try:
    # 50,000 steps for camera
    model.learn(total_timesteps=2**15, callback=callbacks)
except KeyboardInterrupt:
    print("Training interrupted manually.")

# 3. Save Everything
print(f"Saving model to {MODEL_PATH}...")
model.save(MODEL_PATH)

print(f"Saving stats to {STATS_PATH}...")
env.save(STATS_PATH)

print(f"Saving replay buffer to {BUFFER_PATH}...")
model.save_replay_buffer(BUFFER_PATH)

print("Done.")