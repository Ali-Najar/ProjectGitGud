from envs.act_env import ActEnv , QLearningAgent
import numpy as np
import address
import pyMeow as pm
act_env = ActEnv('Q_table_dark_souls_dodge_v1.npy',attack_max=2,speed=1)

agent = QLearningAgent(act_env.observation_space.shape, act_env.action_space,epsilon=1 , epsilon_decay=0.0)
agent.q_table = np.load('Q_table_dark_souls_act_v1.npy')

state = act_env.reset()

while True:  # Run for a certain number of steps
    action = agent.choose_action(state)
    next_state, reward, done, truncated, _ = act_env.step(action)
    if done:
        state = act_env.reset()
        break
    if pm.r_int(address.process,address.Address.iudex_hp) < 570:
        break

act_env.close()