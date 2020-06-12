import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sys import path; path.append("..")
from analyze_random import seaborn_heatmap

env = gym.make('CartPole-v0')

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

def eval_performance(agent, epochs, frames=250, viz=False):

    rewards = []
    for ep in range(epochs):
        obs = [env.reset()]
        total_reward = 0
        terminal = False
        ep_acts = []

        for f in range(frames):
            if ep % (epochs//5) == 0:
                if viz: env.render()
            action = agent.act(obs[-1], time_t=f)
            observation, reward, terminal, info = env.step(action)
            obs.append(observation)
            total_reward += reward
            ep_acts.append(action)

            reward = total_reward

            if reward < 20: # if less than random average
                reward = 0
            
            if not terminal:
                agent.update_policy(action, obs[-2], obs[-1], reward)
            else:
                reward = 0
                agent.update_policy(action, obs[-2], obs[-1], reward, ep_acts)
                break
            
        if ep % (epochs//5) == 0:
            print("Terminal ep:", ep,
                      "\tep_rwd:", total_reward,
                      f"\tlr:{round(agent.lr, 2)}",
                      "\tP(r_act):", round(agent.rand_act_prob, 2))
        
        rewards.append(total_reward)
        
    df = pd.DataFrame({'epoch':np.arange(epochs), 'reward':rewards})
    df['max_reward'] = df[['reward']].rolling(100).max()
    df['mean_reward'] = df[['reward']].rolling(25).mean()
    df['min_reward'] = df[['reward']].rolling(100).min()
    
    df = df.set_index('epoch')
    print(df)
    df.plot()

    plt.ylabel('Episode Reward')
    plt.show()

    # Plot observation axis for Q table
    ax = 0
    q_axis = agent.Q[ax]
    print('q axis', q_axis)
    q_len = len(q_axis)
    q_pivot = {}
    q_pivot['q_value'] = np.concatenate([q_axis[:, 0], q_axis[:, 1]])
    q_pivot[f'obs_{ax}'] = np.concatenate([np.arange(q_len),
                                                                        np.arange(q_len)])
    q_pivot['action'] = np.concatenate([np.zeros(q_len),
                                                                 np.ones(q_len)])
    q_pivot = pd.DataFrame(q_pivot)
    seaborn_heatmap(q_pivot, xyz=('action', f'obs_{ax}', 'q_value'))
    
    return df

class q_learning():
    def __init__(self, env):
        # env.observation_space.high
        # --> [max(val) for val in obs_space]
        # env.observation_space.low
        # --> [min(val) for val in obs_space]

        # model parameters
        digitize_max = 75 #800 # digitizer maps obs to rng(0, digi_max)
        self.gamma = 6.9e-1 # discounted future reward
        
        self.decay = 1 - (2e-5)#1 - (1e-5) # decay factor for below params
        self.rand_act_prob = 1 # 1 # Initial P(random action)
        self.act_prob_min = 0.001 #0.08 #0.15
        self.lr = 2e-2 # 1e-1 # Initial learning rate
        self.lr_min = 0.5e-2 #.5e-2
        # env options
        obs_max = env.observation_space.high
        obs_min = env.observation_space.low
        self.act_len = env.action_space.n
        obs_len = len(obs_min)
        init_limits = [1, 1]

        
        self.best_acts = []

        # Determine best num_bins for environment
        # num_bins  -->  Divisor on X in sigmoid_array
        eval_fx = lambda X, n_b: self.sigmoid_array(X, n_b, digitize_max)        
        num_bins = 2
        '''
        for n_b in range(1, 100, 1):
            if min(eval_fx(obs_max, n_b)) == digitize_max:
                num_bins = n_b
                break
        '''
        print("Determined num bins:", num_bins)
            
        # Digitizer converts array of observations into discrete obs    
        fx = lambda X: self.sigmoid_array(X, num_bins, digitize_max)
        self.digitize = fx


        print("Discrete window:\n\tmax:", fx(obs_max))
        print("\tmin:", fx(obs_min))

        # Q table size is (n, m, k)
        # where: (n=obs_len, m=digi_max, k=act_len)
        #      := tensor(obs_len, [digi_max, act_len])
        q_table_size = (obs_len, digitize_max + 1, self.act_len)
        print("Q_table shape:", q_table_size)
        #self.Q = np.random.uniform(init_limits[0], init_limits[1],
        #                          size=(q_table_size))
        self.Q = np.ones((q_table_size))
        print("\tNumber of values:", len(self.Q.flatten()))

    def sigmoid_array(self, X, num_bins = 20, upper_lim = 10):
        X = upper_lim / (1 + np.exp( -X /num_bins))
        X = tuple(map(lambda x: int(x), X))
        return X

    def get_action_q_values(self, obs):
        acts = np.zeros((self.act_len))
        obs = self.digitize(obs)
        for idx, q_table in enumerate(self.Q):
            for a in range(self.act_len):
                # Normalize contribution for each dim in obs
                acts[a] += self.Q[idx][obs[idx]][a] / len(self.Q)
        return acts

    def set_action_q_values(self, obs, action, q_updated):
        obs = self.digitize(obs)
        for idx in range(len(self.Q)):
            self.Q[idx][obs[idx]][action] = q_updated
            #acts = self.Q[idx][obs[idx]]
            # Normalize q_values to one
            #self.Q[idx][obs[idx]]= acts / sum(acts)
            
    def act(self, obs, time_t=0):
        
        
        
        if np.random.random() < self.rand_act_prob:
            '''
            if len(self.best_acts) > 100: # better than random
                if time_t > len(self.best_acts)/3:
                    if time_t < (2*len(self.best_acts))/3: # only use some of best acts
                        return self.best_acts[time_t] #np.argmax(self.get_action_q_values(obs))
            '''
            return np.random.randint(self.act_len)
        
        
        return np.argmax(self.get_action_q_values(obs))

    def update_policy(self, action, state, next_state, reward, ep_acts=[]):
        next_reward = np.max(self.get_action_q_values(next_state))
        state_value = self.get_action_q_values(next_state)[action]
        
        q_updated = self.lr * (reward + (self.gamma * next_reward))
        q_updated += state_value * (1 - self.lr)
        
        self.set_action_q_values(state, action, q_updated)

        if (2*len(ep_acts))/3 > len(self.best_acts)/2:
            self.best_acts = ep_acts
        
        if self.rand_act_prob > self.act_prob_min:
            self.rand_act_prob *= self.decay
        if self.lr > self.lr_min:
            self.lr *= self.decay

    
    
if __name__ == "__main__":
    epochs = 1000 #500 * 1000
    agent = q_learning(env)
    df = eval_performance(agent, epochs, viz=False)
    

    # Mean reward is 20.5 from q_learning after init, no training, 100ep
