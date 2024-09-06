import player_xyz as pxyz
pxyz.teleport_to_boss()

import numpy as np
import gymnasium as gym
import actions.cam_actions as cam_actions
import envs.cam_env as cam_env
import address
import pyMeow as pm
import pydirectinput as pdir
import time
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv,VecNormalize
import random

process = address.process


class RewardCallback(BaseCallback):
    def __init__(self, env, verbose=0):
        super(RewardCallback, self).__init__(verbose)
        self.rewards = []
        self.normalized_rewards = []
        self.total_steps = 0
        self.env = env
        self.end_flag = False
    def _on_step(self) -> bool:
        # Extract the reward from the environment
        reward = self.env.get_original_reward()
        normalized_reward = self.locals['rewards']
        self.rewards.append(reward)
        self.normalized_rewards.append(normalized_reward)
        self.total_steps += 1

        # Calculate the average reward
        average_reward = sum(self.rewards) / len(self.rewards)
        average_normalized_reward = sum(self.normalized_rewards) / len(self.normalized_rewards)

        enemy_action = address.read_string(process,address.Address.iudex_animation_name)
        player_action = address.read_string(process,address.Address.player_animation_name)
        gamma = self.env.gamma
        # Log the current step and average reward
        obs = self.locals['obs_tensor']
        print(60*'=')
        print(f"Step: {self.total_steps}, Current Reward: {reward} | {normalized_reward} ,  Average Reward: {average_reward} | {average_normalized_reward} , Gamma: {gamma}")
        print(f"Player action: {player_action} , Enemy Action: {enemy_action} , Obs: {obs}")
        if self.locals.get("dones", [False])[0]:
            print("Character died. Stopping training.")
            self.end_flag = True
            return False
        return True

class CamEnvTrain(cam_env.CamEnv):
    def __init__(self , boss_name="iudex", speed = 1):
        super(CamEnvTrain, self).__init__(boss_name="iudex", speed = 1)
        self.loc = [141.87632751464844,-68.88593292236328,593.8687133789062]
        self.episode_length = 30
    
    def random_point_2d(self,x0, z0, max_distance):
        # Generate a random distance
        distance = random.uniform(0, max_distance)

        # Generate a random angle
        angle = random.uniform(0, 2 * np.pi)

        # Convert polar coordinates to Cartesian coordinates
        x = x0 + distance * np.cos(angle)
        z = z0 + distance * np.sin(angle)

        return (x,z)

    def step(self , action):
        reward = 0
        done = False
        info = {}
        if pm.r_int(process , address.Address.hp)<=0:
            done = True
        pm.w_int(process , address.Address.hp , pm.r_int(process , address.Address.max_hp))           
        if self.episode_length == 0:
            x ,z = self.random_point_2d(self.loc[0],self.loc[2],10)
            pm.w_float(process , address.Address.X , x)
            pm.w_float(process , address.Address.Z , z)
            pm.w_float(process , address.Address.Y , self.loc[1])
            self.episode_length = 30
        self.episode_length -= 1
        if action < 4:
            pdir.keyDown(cam_actions.Camera[action][0])
        time.sleep(0.15)
        self.release_key()
        self.reset()
        angle = self.state[6]
        if angle < 0.6:
            reward += 10
        else:
            reward -= 10*angle
        if angle < 0.5 and len(cam_actions.Camera[action])>0:
            reward -= 10
        truncated = False
        print(cam_actions.Camera[action] , angle)
        return self.read_from_memory(), reward, done, truncated , info

env = CamEnvTrain(speed=1)

env = DummyVecEnv([lambda: CamEnvTrain()])

# normalized_env = VecNormalize(env, norm_obs=True, norm_reward=True, gamma=0.95)

normalized_env = VecNormalize.load("vec_normalize_stats_cam_isolated_v2.pkl", env)

custom_callback = RewardCallback(env= normalized_env,verbose=1)

ent_coef = 0.004

model = PPO.load("ppo_dark_souls_cam_isolated_v2", env=normalized_env)

params = model.get_parameters()

model = PPO('MlpPolicy',learning_rate=0.0003,n_steps=512 , batch_size=64 ,  n_epochs=15 , ent_coef=ent_coef ,env = normalized_env, verbose=1)

model.set_parameters(params)


model.learn(total_timesteps=4096 , callback=custom_callback)

normalized_env.save("vec_normalize_stats_cam_isolated_v2.pkl")

model.save("ppo_dark_souls_cam_isolated_v2")