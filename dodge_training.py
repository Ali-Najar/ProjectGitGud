import numpy as np
import gymnasium as gym
import actions.act_actions as act_actions
import envs.dodge_env as dodge_env
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
        print(f"Step: {self.total_steps}, Current Reward: {reward} | {normalized_reward} ,  Average Reward: {average_reward} | {average_normalized_reward} , Gamma: {gamma}")
        print(f"Player action: {player_action} , Enemy Action: {enemy_action} , Obs: {obs}")
        print(60*'=')
        if self.locals.get("dones", [False])[0]:
            print("Character died. Stopping training.")
            self.end_flag = True
            return False
        return True


class DodgeEnvTrain(dodge_env.DodgeEnv):
    def __init__(self , boss_name="iudex", speed = 1):
        super(DodgeEnvTrain, self).__init__()
        self.speed = speed
    
    def step(self , action):
        reward = 0
        done = False
        info = {}
        if 'death' in address.read_string(process , address.Address.player_animation_name).lower():
            done = True
        if pm.r_int(process , address.Address.iudex_hp)<=0 or pm.r_int(process , address.Address.hp)<=0:
            done = True
        prev_attacking = self.state[0]
        prev_hp = pm.r_int(process , address.Address.hp)
        if action == 0:
            pdir.press(act_actions.Dodge[action][0])
            time.sleep(max(0,pm.r_float(process,address.Address.player_max_animation_time)-0.2))
        hp = pm.r_int(process , address.Address.hp)
        if prev_attacking == 0 and action == 1:
            reward += 1
        if prev_attacking == 1 and action == 0:
            reward += 3
            if hp < prev_hp:
                reward -= 2
            else:
                print("FDS")
                reward += 5
        if prev_attacking == 0 and action == 0:
            reward -= 3
        if hp < 200:
            pm.w_int(process,address.Address.hp,pm.r_int(process,address.Address.max_hp))
        print(f'action: {act_actions.Dodge[action]}')
        truncated = False
        return self.read_from_memory(), reward, done, truncated , info
    
env = DodgeEnvTrain(speed=1)

env = DummyVecEnv([lambda: DodgeEnvTrain()])

# normalized_env = VecNormalize(env, norm_obs=True, norm_reward=True, gamma=0.95)

normalized_env = VecNormalize.load("vec_normalize_stats_dodge_v1.pkl", env)

custom_callback = RewardCallback(env= normalized_env,verbose=1)

ent_coef = 0.02

model = PPO.load("ppo_dark_souls_dodge_v1", env=normalized_env)

params = model.get_parameters()

model = PPO('MlpPolicy',learning_rate=0.0005,n_steps=512 , batch_size=64 ,  n_epochs=20 , ent_coef=ent_coef ,env = normalized_env, verbose=1)

model.set_parameters(params)

model.learn(total_timesteps=4096 , callback=custom_callback)

normalized_env.save("vec_normalize_stats_dodge_v1.pkl")

model.save("ppo_dark_souls_dodge_v1")