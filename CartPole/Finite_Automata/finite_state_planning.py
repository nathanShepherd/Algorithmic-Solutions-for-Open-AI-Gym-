import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sb

from sys import path; path.append("..")
from analyze_random import seaborn_heatmap

env = gym.make('CartPole-v0')
'''
    observation                 Min         Max
    0	Cart Position             -4.8            4.8
    1	Cart Velocity             -Inf            Inf
    2	Pole Angle                 -24           24
    3	Pole Velocity At Tip      -Inf            Inf
'''

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
                rewards.append(total_reward)
                
                agent.terminal(obs, total_reward, ep_acts)
                agent.update_policy(action, obs[-2], obs[-1], reward, ep_acts)
                break
            
        if ep % (epochs//5) == 0:
            print("Terminal ep:", ep,
                      "\tep_rwd:", total_reward,
                      #f"\tlr:{round(agent.lr, 2)}",
                      "\tP(r_act):", round(agent.rand_act_prob, 2))
        
        
        
    df = pd.DataFrame({'epoch':np.arange(epochs), 'reward':rewards})
    df['max_reward'] = df[['reward']].rolling(100).max()
    df['mean_reward'] = df[['reward']].rolling(25).mean()
    df['min_reward'] = df[['reward']].rolling(100).min()
    
    df = df.set_index('epoch')
    print(df)
    df.plot()

    plt.ylabel('Episode Reward')
    plt.show()

    # Plot learned parameters of model
    ax = 1
    q_axis = agent.delta[ax]
    print('q axis', q_axis)
    q_len = len(q_axis)
    q_pivot = {}
    q_pivot['next_state'] = np.concatenate([ q_axis[:, 0],   q_axis[:, 1] ])
    q_pivot[f'state_{ax}'] = np.concatenate([ np.arange(q_len), np.arange(q_len) ])
    q_pivot['action'] = np.concatenate([ np.zeros(q_len),    np.ones(q_len) ])
    q_pivot = pd.DataFrame(q_pivot)
    #seaborn_heatmap(q_pivot, xyz=('action', f'state_{ax}', 'next_state'))

    # Plot learned final states
    obs_len = len(agent.final_states[0])
    obs_max = len(agent.final_states)
    f_states_counts = np.concatenate(agent.final_states)
    obs_axes = [ idx for idx in range(obs_len)]
    obs_axes = [np.ones(obs_max) * i for i in obs_axes]
    obs_axes = np.concatenate(obs_axes)
    #print('f_states', f_states)
    #print('obs_axes', obs_axes)
    end_states = np.concatenate([np.arange(obs_max) for i in range(obs_len)])
    
    f_pivot = pd.DataFrame({'obs_axis':obs_axes,
                                                  'final_state':end_states,
                                                  'state_count':f_states_counts})
    #sb.catplot(x='obs_axis', y='final_state', data=f_states)
    #seaborn_heatmap(f_pivot, xyz=('obs_axis', 'final_state', 'state_count'))

    # TODO: visualize plans, and plan value
    
    return df

class q_learning():
    def __init__(self, env):
        # env.observation_space.high
        # --> [max(val) for val in obs_space]
        # env.observation_space.low
        # --> [min(val) for val in obs_space]

        # model parameters
        self.rand_act_prob = 1 # 1 # Initial P(random action)
        self.act_prob_min = 0.001 #0.08 #0.15
        
        # env options
        obs_max = env.observation_space.high
        obs_min = env.observation_space.low
        self.act_len = env.action_space.n
        obs_len = len(obs_min)
        init_limits = [1, 1]
        self.best_acts = []

        # Determine best num_bins for environment
        # num_bins  -->  Divisor on X in sigmoid_array
        digitize_max = 20 #800 # 
        num_bins = digitize_max// 2
            
        # Digitize() maps obs in observation to rng(0, digi_max)
        self.digitize = lambda X: self.sigmoid_array(X, num_bins, digitize_max)
        digitize_max += 1 # Correction so that max obs is included in size ranges

        print("Discrete window:\n\tmax:", self.digitize(obs_max))
        print("\tmin:", self.digitize(obs_min))

        self.delta = np.zeros((obs_len, digitize_max, self.act_len))
        self.final_states = np.zeros((digitize_max, obs_len))

        # Plans are a policy of actions from start to end states
        plans_shape = (obs_len, digitize_max, digitize_max)
        self.plans = np.zeros(plans_shape)

        # Plans point to a str hash, convert back to actions list with acts_dict
        self.acts_dict = {}

        # Plans value associates reward for following a plan (via hash(str(acts)) )
        self.plan_value = {}


    def sigmoid_array(self, X, num_bins = 20, upper_lim = 10):
        X = upper_lim / (1 + np.exp( -num_bins * X ))
        X = tuple(map(lambda x: int(x), X))
        return X

    def get_action_values(self, obs):
        acts = np.zeros((self.act_len))
        obs = self.digitize(obs)
        return acts

    def act(self, obs, time_t=0):
        axis = 3
        obs = self.digitize(obs)[axis]
        
        if len(self.acts_dict) < 1 or sum(self.plans[axis][obs]) == 0:
            return np.random.randint(self.act_len)

        #import pdb; pdb.set_trace()

        plan_values = []
        # TODO: change plans to a smaller interval
        # TODO: try a walk from [this_obs, action] to final states
        for acts_hash in self.plans[axis][obs]:
            # TODO: incorporate final states into plan value
            if acts_hash in self.acts_dict:
                hash_to_value = [acts_hash, self.plan_value[acts_hash]]
                plan_values.append(np.array(hash_to_value))
            else:
                plan_values.append(np.array([-1, -1]))

        plan_values = np.array(plan_values)
        # If no plan of value, take random action
        if sum(plan_values[:, 1]) == -1 * len(plan_values):
            return np.random.randint(self.act_len)
        
        choice = np.argmax(plan_values[:, 1])
        plan = self.acts_dict[plan_values[choice][0]]
        action = plan[1]
        return action

    def set_final_state(self, obs_d):
        # obs_d = digitized observation
        for d_idx, obs in enumerate(obs_d):
            
            self.final_states[obs][d_idx] += 1

    def remap_transition(self, action, state_d, next_state_d):
        for d_idx, obs in enumerate(state_d):
            self.delta[d_idx][obs][action] = next_state_d[d_idx]

    def update_policy(self, action, state, next_state, reward, ep_acts=[]):
        state = self.digitize(state)
        next_state = self.digitize(next_state)

        self.remap_transition(action, state, next_state)

    def terminal(self, ep_obs, total_reward, ep_acts):
        
        for frame in range(len(ep_obs)):
                ep_obs[frame] = self.digitize(ep_obs[frame])
                
        initial_state = ep_obs[0]
        final_state = ep_obs[-1]
        
        self.set_final_state(final_state)

        # Update plan
        acts_hash = hash(str(ep_acts))
        for d_idx in range(len(ep_obs[0])):
            start = initial_state[d_idx]
            end = final_state[d_idx]
            
            self.plans[d_idx][start][end] = acts_hash
            self.acts_dict[acts_hash] = ep_acts
            self.plan_value[acts_hash] = total_reward
                
            
    
    
if __name__ == "__main__":
    epochs = 10 * 1000
    agent = q_learning(env)
    df = eval_performance(agent, epochs, viz=False)
    

    # Mean reward is 20.5 from q_learning after init, no training, 100ep
