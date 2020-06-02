import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

env = gym.make('LunarLander-v2')

def play_random(viz=True):
    obs = [env.reset()]
    total_reward = 0
    terminal = False

    while not terminal:
        if viz: env.render()
        action = env.action_space.sample()
        observation, reward, terminal, info = env.step(action)
        obs.append(observation)
        total_reward += reward
        
    return obs, total_reward

def episodes(epochs, viz=False):

    frames, rewards = [],[]
    for ep in range(epochs):
        X = play_random(viz)
        frames.append(X[0])
        rewards.append(X[1])
        
    return {'obs':frames, 'reward':rewards}

def plot_observations(epochs=2):
    df = pd.DataFrame(episodes(epochs))
    obs_matrix = np.array(df['obs'].values)
    
    '''
    for i in range(epochs):
        obs_matrix = np.array(df['obs'].values[i])
        print(obs_matrix.shape)
        omat = pd.DataFrame(obs_matrix)
        print(omat.head())
        omat.plot()
    plt.show()
    '''
    
if __name__ == "__main__":
    plot_observations()
    #df.plot()
    #plt.show()
