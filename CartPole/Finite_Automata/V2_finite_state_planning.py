import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mode

import seaborn as sb

from sys import path; path.append("..")
from analyze_random import seaborn_heatmap
sb.set_style('dark')
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

def eval_performance(agent, epochs,
                                         frames=250, #breaks indexing
                                         viz=False):

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

            #reward = total_reward
            

            #if reward < 20: # if less than random average
            #   reward = 0
            
            if not terminal:
                agent.update_policy(action, obs[-2], obs[-1], reward)
            else:
                
                reward = 0
                rewards.append(total_reward)
                agent.terminal(obs, reward, ep_acts)
                
                agent.update_policy(action, obs[-2], obs[-1], reward)
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
    #print(df)
    df.plot()

    plt.ylabel('Episode Reward')
    plt.show()

    ### Plot learned parameters of model ###
    ax = 3
    q_axis = agent.delta[ax]
    #print('q axis', q_axis)
    q_len = len(q_axis)
    q_pivot = {}
    q_pivot['next_state'] = np.concatenate([ q_axis[:, 0],   q_axis[:, 1] ])
    q_pivot[f'state_{ax}'] = np.concatenate([ np.arange(q_len), np.arange(q_len) ])
    q_pivot['action'] = np.concatenate([ np.zeros(q_len),    np.ones(q_len) ])
    q_pivot = pd.DataFrame(q_pivot)
    #seaborn_heatmap(q_pivot, xyz=('next_state', f'state_{ax}', 'action'), annotate=True)

    # Plot learned final states
    obs_len = len(agent.final_states[0])
    obs_max = len(agent.final_states)
    f_states_counts = np.concatenate([counts/max(counts) for counts in agent.final_states])
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

    # Visualize plans and plan value
    ax = 1
    q_axis = agent.plans[ax]
    print('q axis', q_axis)
    q_len = len(q_axis)
    q_pivot = {}
    
    q_pivot['end'] =  np.concatenate([np.arange(q_len)] * q_len) 
    q_pivot[f'start_state_{ax}'] = np.concatenate(np.array([[np.ones(q_len).astype(int) * i] for i in range(q_len)]))
    q_pivot[f'start_state_{ax}'] = np.concatenate(q_pivot[f'start_state_{ax}'])
    
    q_pivot['plan_value'] = []
    for start, end in list(zip(q_pivot[f'start_state_{ax}'], q_pivot['end'] )):
        plan_hash = q_axis[int(start)][end]
        if plan_hash in agent.plan_value:
            q_pivot['plan_value'].append(agent.plan_value[plan_hash])
            #print('found_plan', start, end, agent.plan_value[plan_hash])
        else:
            q_pivot['plan_value'].append(0)
            
    q_pivot = pd.DataFrame(q_pivot)
    seaborn_heatmap(q_pivot, xyz=(f'start_state_{ax}', 'end', 'plan_value'))
    
    return df

class q_learning():
    def __init__(self, env):
        # env.observation_space.high
        # --> [max(val) for val in obs_space]
        # env.observation_space.low
        # --> [min(val) for val in obs_space]

        # model parameters
        self.rand_act_prob = 0.5 # .5 # 1 # Initial P(random action)
        self.act_prob_min = 0 #.00001 #0.08 #0.15
        self.decay = 1 - (2e-4)
        
        # env options
        obs_max = env.observation_space.high
        obs_min = env.observation_space.low
        self.act_len = env.action_space.n
        obs_len = len(obs_min)
        init_limits = [1, 1]
        self.best_acts = []

        # Digitize() maps obs in observation to rng(0, digi_max)
        # num_bins  -->  Divisor on X in sigmoid_array
        self.digitize_max = 50 #800 # 
        self.num_bins = self.digitize_max// 2
            
        
        # Correction so that max obs is included in size ranges
        digitize_lim =self.digitize_max + 1 

        print("Discrete window:\n\tmax:", self.digitize(obs_max))
        print("\tmin:", self.digitize(obs_min))

        self.delta = np.zeros((obs_len, digitize_lim, self.act_len)).astype(int)
        self.final_states = np.zeros((digitize_lim, obs_len))

        # Plans are a policy of actions from start to end states
        plans_shape = (obs_len, digitize_lim, digitize_lim)
        self.plans = np.zeros(plans_shape)

        # Plans point to a str hash, convert back to actions list with acts_dict
        self.acts_dict = {}

        # Plans value associates reward for following a plan (via hash(str(acts)) )
        self.plan_value = {}
        self.max_plan_value = np.zeros((obs_len)) # TODO: deprioritize max values

    
    def sigmoid_array(self, X, num_bins = 20, upper_lim = 10):
        X = upper_lim / (1 + np.exp( -num_bins * X ))
        X = tuple(map(lambda x: int(x), X))
        return X
    
    def digitize(self, X):
        return self.sigmoid_array(X, self.num_bins, self.digitize_max)
    
    def get_action_values(self, obs):
        acts = np.zeros((self.act_len))
        obs = self.digitize(obs)
        return acts

    def next_state_action_values(self, d_idx, obs):
        plan_values = []
        for acts_hash in self.plans[d_idx][obs[d_idx]]:
            if acts_hash in self.acts_dict:
                hash_to_value = [acts_hash, self.plan_value[acts_hash]]                    
                plan_values.append(np.array(hash_to_value))
            #else:
            #   plan_values.append(np.array([-1, -1]))

        plan_values = np.array(plan_values)
        # If no plan of value, take random action
        
        
        return plan_values

    def act(self, obs, time_t=0, simulated=False):
        d_idx = 3 # best at 3 (pole velocity at tip), fails with other idxs
        
        if not simulated:
            obs = self.digitize(obs) # [axis]
        
        if np.random.random() < self.rand_act_prob:
            return np.random.randint(self.act_len)

        
        # TODO: try a walk from [this_obs, action] to final states
        # TODO: try early prioritization of plan_values with no value
        #for d_idx in range(len(obs[:2])):

        # Take random action if no plans exist or no plan exists for this obs
        if len(self.acts_dict) < 1 or sum(self.plans[d_idx][obs[d_idx]]) == 0:
            return np.random.randint(self.act_len)

            ## Take action that does not result in end state
            # final_states[next_states[action]][idx] > 0 if obs and action result in final state

        suggested_acts = []
        for idx in range(len(obs)):
            plan_value = self.next_state_action_values(idx, obs)
        
            if len(plan_value) == 0: # or max(plan_values[:, 1]) < 100:
                return np.random.randint(self.act_len)
        
            choice = np.argmax(plan_value[:, 1])
            choice_max = max(plan_value[:, 1])
            suggested_acts.append(np.array([self.acts_dict[plan_value[choice][0]], choice_max]))

        choice_act = mode(np.array(suggested_acts)[:, 0]).mode.astype(int)[0]
        #print(choice_act)
        '''
        plan_values = self.next_state_action_values(d_idx, obs)

        ### Adding new obs axis
        plan_v_idx0 = self.next_state_action_values(2, obs)
        
        
        if len(plan_values) == 0 or len(plan_v_idx0) == 0: # or max(plan_values[:, 1]) < 100:
            return np.random.randint(self.act_len)
        
        choice = np.argmax(plan_values[:, 1])
        choice_max = max(plan_values[:, 1])
        choice_act = self.acts_dict[plan_values[choice][0]]

        plan_v0_max = max(plan_v_idx0[:, 1])
        if plan_v0_max > choice_max:
            choice = np.argmax(plan_v_idx0[:, 1])
            choice_act = self.acts_dict[plan_v_idx0[choice][0]]
        '''
        # Select hash index (0) from chosen plan value
        return choice_act

    def set_final_state(self, obs_d):
        # obs_d = digitized observation
        for d_idx, obs in enumerate(obs_d):
            
            self.final_states[obs][d_idx] += 1

    def remap_transition(self, action, state_d, next_state_d):
        for d_idx, obs in enumerate(state_d):
            self.delta[d_idx][obs][action] = next_state_d[d_idx]

    def state_hash(self, start, end, d_idx, action):
        return float(hash(f'{start}|{end}|{d_idx}|{action}'))

    def update_policy(self, action, state, next_state, reward, ep_acts=[]):
        #future_state = next_state
        state = self.digitize(np.array(state))
        next_state = self.digitize(np.array(next_state))

        self.remap_transition(action, state, next_state)

        if self.rand_act_prob > self.act_prob_min:
            self.rand_act_prob *= self.decay
        if self.rand_act_prob <= self.act_prob_min:
            self.rand_act_prob = 0
        # TODO: determine [action, state] value and [state] value
        #               optimal policy --> most expected return for any state
        #               optimal q_value --> most expected return for any [action, state]
        #               q_value[s, a] = E[R(t+ 1) + future_decay(max(Q[state, action])]
        #               state_value = E[R(t+ 1) + future_decay(max(state_value)]
        # Update plans
        # Follow policy and determine if reaches final state
        reaches_terminal = False
        for d_idx in range(len(state)):
            future_state = next_state
            for frame in range(5):
                future_action = self.act(future_state, simulated=True)
                #print(future_state)
                future_state = self.delta[:, future_state[d_idx], future_action]
            
            if self.final_states[future_state[d_idx]][d_idx] > 0:
                reaches_terminal = True

        #reaches_terminal = False ########## NOte
        
        for d_idx in range(len(state)):            
            start, end = state[d_idx], next_state[d_idx]
            #if reward == 0:
             #   print(f'Plan Update| start:{start},\tend{end},',
             #             f'\tax{d_idx},\tact:{action}\tvalue{reward}')
            
            h_str = self.state_hash(start, end, d_idx, action)
            self.plans[d_idx][start][end] = h_str
            self.acts_dict[h_str] = action
            if h_str in self.plan_value:
                #print(f"New plan|\tprev:{self.plan_value[h_str]}\tnew:{reward}")
                # TODO scale by final count of state next if final state
                future_state_value = self.next_state_action_values(d_idx, next_state)#next_state)
                if len(future_state_value) > 0:
                    future_state_value = future_state_value[np.argmax(future_state_value[:, 1])][0]
                    future_state_value = self.plan_value[future_state_value]
                else:
                    future_state_value = 0
                    
                if self.final_states[state[d_idx]][d_idx] > 0: #[next_state[d_idx]][d_idx] > 0:
                    future_state_value -=  0.1 * self.final_states[state[d_idx]][d_idx]
                    #future_state_value /=  self.final_states[state[d_idx]][d_idx]

                # Determine if final state is likely given next state
                future_state_a0 = self.delta[d_idx][next_state[d_idx]][0]
                future_state_a1 = self.delta[d_idx][next_state[d_idx]][1]
                if self.final_states[future_state_a0][d_idx] > 0 or self.final_states[future_state_a1][d_idx] > 0:
                    #future_state_value /= self.final_states[future_state_a0][d_idx] + self.final_states[future_state_a1][d_idx]
                    future_state_value /= self.final_states[future_state_a0][d_idx] + self.final_states[future_state_a1][d_idx]
                                                    
                #if self.final_states[state[d_idx]][d_idx] > 0:
                #    reward /= self.final_states[state[d_idx]][d_idx]

                if self.final_states[future_state[d_idx]][d_idx] > 0:
                    future_state_value = 0
                
                if reaches_terminal:
                    future_state_value = 0
                    
                reward += .1 * future_state_value
                if self.plan_value[h_str] > 0:
                    self.plan_value[h_str] += (reward * .01) #0.01) #/ (self.plan_value[h_str] ) #* self.decay #+= reward

                    if self.plan_value[h_str] > 0.7 * self.max_plan_value[d_idx]:
                        self.plan_value[h_str] *= 0.5
                        
                    if self.plan_value[h_str] > self.max_plan_value[d_idx]:
                        self.max_plan_value[d_idx] = 0.7 * self.plan_value[h_str]
                        
                else:
                    self.plan_value[h_str] = future_state_value
            else:
                self.plan_value[h_str] = reward

    def terminal(self, ep_obs, total_reward, ep_acts):
        
        for frame in range(len(ep_obs)):
                ep_obs[frame] = self.digitize(ep_obs[frame])
                
        #initial_state = ep_obs[0]
        
        
        
        self.set_final_state(ep_obs[-1])
        #self.set_final_state(ep_obs[-2])
        #self.set_final_state(ep_obs[-3])
        #self.set_final_state(ep_obs[-4])
        '''
        # Update plan
        acts_hash = hash(str(ep_acts))
        for d_idx in range(len(ep_obs[0])):
            start = initial_state[d_idx]
            end = final_state[d_idx]
            
            self.plans[d_idx][start][end] = acts_hash
            self.acts_dict[acts_hash] = ep_acts
            self.plan_value[acts_hash] = total_reward
        '''      
            
    
    
if __name__ == "__main__":
    np.random.seed(99)
    epochs = 3 * 100
    agent = q_learning(env)
    df = eval_performance(agent, epochs, viz=True)
    

    # Mean reward is 20.5 from q_learning after init, no training, 100ep
