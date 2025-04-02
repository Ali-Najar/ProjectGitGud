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
import torch
import torch.nn as nn
import torch.optim as optim
import keyboard
from envs.dodge_env import QLearningAgent


process = address.process

# class RewardCallback(BaseCallback):
#     def __init__(self, env, verbose=0):
#         super(RewardCallback, self).__init__(verbose)
#         self.rewards = []
#         self.normalized_rewards = []
#         self.total_steps = 0
#         self.env = env
#         self.end_flag = False
        
#     def _on_step(self) -> bool:
#         # Extract the reward from the environment
#         reward = self.env.get_original_reward()
#         normalized_reward = self.locals['rewards']
#         self.rewards.append(reward)
#         self.normalized_rewards.append(normalized_reward)
#         self.total_steps += 1

#         # Calculate the average reward
#         average_reward = sum(self.rewards) / len(self.rewards)
#         average_normalized_reward = sum(self.normalized_rewards) / len(self.normalized_rewards)

#         enemy_action = address.read_string(process,address.Address.iudex_animation_name)
#         player_action = address.read_string(process,address.Address.player_animation_name)
#         gamma = self.env.gamma
#         # Log the current step and average reward
#         obs = self.locals['obs_tensor']
#         with torch.no_grad():
#             action_dist = model.policy.get_distribution(obs)
#             print("Logits for each action:", action_dist.distribution.probs)
#         print(f"Step: {self.total_steps}, Current Reward: {reward} | {normalized_reward} ,  Average Reward: {average_reward} | {average_normalized_reward} , Gamma: {gamma}")
#         print(f"Player action: {player_action} , Enemy Action: {enemy_action} , Obs: {obs}")
#         print(60*'=')
#         if self.locals.get("dones", [False])[0]:
#             print("Character died. Stopping training.")
#             self.end_flag = True
#             return False
#         return True


class DodgeEnvTrain(dodge_env.DodgeEnv):
    def __init__(self , boss_name="iudex", speed = 1):
        super(DodgeEnvTrain, self).__init__()
        self.speed = speed
        self.prev_action = None
    
    def step(self , action):
        reward = 0
        done = False
        info = {}
        if 'death' in address.read_string(process , address.Address.player_animation_name).lower():
            done = True
        if pm.r_int(process , address.Address.iudex_hp)<=0 or pm.r_int(process , address.Address.hp)<=0:
            done = True
        prev_attacking = self.state[0]
        dodged = self.state[1]
        ratio = self.state[2]
        max_time = self.state[3]
        prev_hp = pm.r_int(process , address.Address.hp)
        if action == 0:
            pdir.keyDown('w')
            pdir.press(act_actions.Dodge[action][0])
            pdir.keyUp('w')
            # time.sleep(max_time-ratio*max_time)
            time.sleep(max(0,pm.r_float(process,address.Address.player_max_animation_time)-1.4))
        else:
            time.sleep(0.1)
        hp = pm.r_int(process , address.Address.hp)
        if prev_attacking == 0 or dodged == 1:
            if action == 0:
                reward -= 6
        if prev_attacking == 1 and dodged == 0:
            if action == 0:
                if hp < prev_hp:
                    reward -= 3
                else:
                    reward += 5
            if action == 1:
                if hp < prev_hp:
                    reward -= 20
        # if prev_attacking == 0 or dodged == 1:
        #     if action == 0:
        #         reward -= 60
        #     if action == 1:
        #         reward += 1
        # if prev_attacking == 1 and dodged == 0:
        #     if ratio > 0.30 and ratio < 0.46:
        #         if action == 0:
        #             reward += 30
        #             if hp < prev_hp:
        #                 reward -= 50
        #             else:
        #                 reward += 30
        #         if action == 1:
        #             reward -= 50
        #     elif ratio > 0.2 and ratio < 0.5:
        #         if action == 0:
        #             reward += 15
        #             if hp < prev_hp:
        #                 reward -= 40
        #             else:
        #                 reward += 15
        #     else:
        #         if action == 0:
        #             reward -= 30
        #         else:
        #             if ratio <0.25:
        #                 reward += 5
        #             else:
        #                 reward += 3
        # if action == 1:
        #     if hp < prev_hp:
        #         reward -= 35
        if action == 0:
            self.action_dodged = True
        if hp < 200:
            pm.w_int(process,address.Address.hp,pm.r_int(process,address.Address.max_hp))
        print(f'action: {act_actions.Dodge[action]}')
        self.reset()
        truncated = False
        return self.read_from_memory(), reward, done, truncated , info
    
env = DodgeEnvTrain(speed=1)

# env = DummyVecEnv([lambda: DodgeEnvTrain()])

# normalized_env = VecNormalize(env, norm_obs=False, norm_reward=False, gamma=0.95)

# normalized_env = VecNormalize.load("vec_normalize_stats_dodge_v8.pkl", env)

# custom_callback = RewardCallback(env= normalized_env,verbose=1)

# ent_coef = 0.02

# model = PPO.load("ppo_dark_souls_dodge_v8", env=normalized_env)

# params = model.get_parameters()

# model = PPO('MlpPolicy',learning_rate=0.1,n_steps=512 , batch_size=128 ,  n_epochs=20 , ent_coef=ent_coef ,env = normalized_env, verbose=1)

# model.set_parameters(params)

# model.learn(total_timesteps=1024*2 , callback=custom_callback)

# Create a custom discretization function for continuous state spaces


# Q-learning agent

# Environment and agent setup
agent = QLearningAgent(env.observation_space.shape, env.action_space,epsilon=0,epsilon_min=0.01 ,epsilon_decay=0.995)
agent.q_table = np.load('Q_table_dark_souls_dodge_v1_phase2.npy')
# Training loop
episodes = 300
steps = 128
pm.w_int(address.process , address.Address.iudex_hp , 560)
for episode in range(episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0

    for _ in range(steps):
        action = agent.choose_action(state)
        # action = get_action()
        next_state, reward, done, truncated, _ = env.step(action)
        agent.learn(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

    print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward}")

env.close()

np.save('Q_table_dark_souls_dodge_v1_phase2', agent.q_table)
# normalized_env.save("vec_normalize_stats_dodge_v8.pkl")
# model.save("A2C_dark_souls_dodge_v1")