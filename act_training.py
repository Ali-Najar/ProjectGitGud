import numpy as np
import gymnasium as gym
import actions.act_actions as act_actions
import envs.act_env as act_env
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


class ActEnvTrain(act_env.ActEnv):
    def __init__(self , boss_name="iudex", speed = 1):
        super(ActEnvTrain, self).__init__()
        self.speed = speed
        self.hp_lock = pm.r_int(process,address.Address.max_hp)
        self.estus_regeneration = 100
    
    def step(self , action):
        reward = 0
        done = False
        info = {}
        if 'death' in address.read_string(process , address.Address.player_animation_name).lower():
            done = True
        if pm.r_int(process , address.Address.iudex_hp)<=0 or pm.r_int(process , address.Address.hp)<=0:
            done = True
        prev_dist = self.state[0]
        prev_estus = self.state[2]
        prev_attacking = self.state[1]
        prev_hp = self.state[4]
        prev_stamina = self.state[3]
        iudex_prev_hp = self.state[5]
        if action > 0 and action < 3:
            pdir.keyDown(act_actions.Actions[action][0])
            # if 'attack' or 'item' in address.read_string(process , address.Address.player_animation_name).lower():
            time.sleep(max(0,pm.r_float(process,address.Address.player_max_animation_time)))
            pdir.keyUp(act_actions.Actions[action][0])
        if action == 0 or action == 3:
            if action != 3 or prev_estus == 1:                
                pdir.press(act_actions.Actions[action][0])
                # if 'roll' in address.read_string(process , address.Address.player_animation_name).lower():
                time.sleep(max(0,pm.r_float(process,address.Address.player_max_animation_time)))
            else:
                time.sleep(1)
        self.reset()
        hp = self.state[4]
        estus = self.state[2]
        iudex_hp = self.state[5]
        if prev_attacking == 1:
            if action == 0:
                if hp < prev_hp:
                    reward += 50
                else:
                    reward += 200
            elif action != 4:
                reward -= 200
        reward += hp - prev_hp
        if prev_attacking == 0: 
            reward += (iudex_prev_hp - iudex_hp)*2
        if prev_dist > 5:
            if action == 2 or action == 1:
                reward -= 100
        if action == 3:
            if prev_estus == 0:
                reward -= 200
            else:
                if prev_hp > 0.6 * self.maximum_hp:
                    reward -= 400
                if prev_hp < 0.6 * self.maximum_hp and prev_attacking == 0:
                    reward += 100
        if iudex_hp < 900 and iudex_prev_hp >= 900:
            reward += 500
        if iudex_hp < 750 and iudex_prev_hp >= 750:
            reward += 500
        if iudex_hp < 600 and iudex_prev_hp >= 600:
            reward += 500
        if prev_stamina < self.maximum_stamina*0.2 and action <=2:
            reward -= 200
        if hp<=150:
            reward -= 300
        if estus == 0:
            self.estus_regeneration -= 1
            if self.estus_regeneration == 0:
                pm.w_int(process, address.Address.estus , 4)
                self.estus_regeneration = 100
        print(f'action: {act_actions.Actions[action]}')
        truncated = False
        if pm.r_int(address.process, address.Address.iudex_hp) < 680:
            pm.w_int(address.process, address.Address.iudex_hp , pm.r_int(address.process, address.Address.iudex_max_hp))
        return self.read_from_memory(), reward, done, truncated , info
    
env = ActEnvTrain(speed=1)

env = DummyVecEnv([lambda: ActEnvTrain()])

# normalized_env = VecNormalize(env, norm_obs=True, norm_reward=True, gamma=0.95)

normalized_env = VecNormalize.load("vec_normalize_stats_act_v2.pkl", env)

custom_callback = RewardCallback(env= normalized_env,verbose=1)

ent_coef = 0.02

model = PPO.load("ppo_dark_souls_act_v2", env=normalized_env)

params = model.get_parameters()

model = PPO('MlpPolicy',learning_rate=0.0005,n_steps=1024 , batch_size=128 ,  n_epochs=20 , ent_coef=ent_coef ,env = normalized_env, verbose=1)

model.set_parameters(params)

model.learn(total_timesteps=8192 , callback=custom_callback)

normalized_env.save("vec_normalize_stats_act_v2.pkl")

model.save("ppo_dark_souls_act_v2")