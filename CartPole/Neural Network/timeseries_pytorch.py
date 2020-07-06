
import gym
import torch
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

from time import time

from nn import NN
from train_nn import train_epoch, evaluate_epoch

from sys import path; path.append("..")
from analyze_random import seaborn_heatmap


env = gym.make('CartPole-v0')
#plt.style.use('bmh')
sb.set_style('darkgrid')


def play_random(viz=True):
    obs = [env.reset()]
    total_reward = []
    frame_actions = [0.5] # init with median
    terminal = False

    while not terminal:
        if viz: env.render()
        action = env.action_space.sample()
        frame_actions.append(action)
        observation, reward, terminal, info = env.step(action)
        obs.append(observation)
        total_reward.append(reward)

    # Append Zero for terminal reward
    total_reward.append(0)
        
    return obs, np.array(total_reward), np.array(frame_actions)

def episodes(epochs, viz=False):

    frames, rewards, f_actions = [],[],[]
    for ep in range(epochs):
        X = play_random(viz)
        frames.append(X[0])
        rewards.append(X[1])
        f_actions.append(X[2])
        
    return {'obs':frames, 'reward':rewards, 'f_actions':f_actions}

    
def save_observations(filename, epochs=1000):
    
    frames_rewards = episodes(epochs)
    df = pd.DataFrame(frames_rewards)
    rewards = df['reward'].values
    observations = df['obs'].values
    actions = df['f_actions'].values

    df_arr  = []
    
    for i, ep_obs in enumerate(observations):
        ep_obs = np.array(ep_obs)
        ep_len = len(ep_obs[:, 0])

        ep_dict = {'timestep':np.arange(ep_len)}
        ep_dict['episode'] = np.ones(ep_len) * i
        ep_dict['reward'] = rewards[i]
        ep_dict['f_actions'] = actions[i]
        
                
        for ax in range(len(ep_obs[0])):
            ep_dict[f'obs_ax{ax}'] = ep_obs[:, ax]

        ep_dict = pd.DataFrame(ep_dict)
        df_arr.append(ep_dict)
        
    #print(df_arr)
    #
    #import pdb; pdb.set_trace()
    
    df = pd.concat(df_arr)

    df.to_csv(filename)
    
    #heatmap(obs_matrix, "Observations by episode")
    
    
def plot_observation_reward(filename):
    df = pd.read_csv(filename)
    seaborn_heatmap(df, xyz=('timestep', 'obs_ax0', 'reward'))
    seaborn_heatmap(df, xyz=('timestep', 'obs_ax1', 'reward'))
    seaborn_heatmap(df, xyz=('timestep', 'obs_ax2', 'reward'))
    seaborn_heatmap(df, xyz=('timestep', 'obs_ax3', 'reward'))

def observe_data(filename):
    df = pd.read_csv(filename)
    #import pdb; pdb.set_trace()
    df.info()
    
    cols = ['reward', 'f_actions', 'timestep']
    print(df[cols].describe())
    #print(df[['timestep']].describe().max())

def compute_batches(inputs, batch_size):
    idx = np.linspace(0, len(inputs), endpoint=False,
                      num=len(inputs)//batch_size).astype(int)
    batches = []
    for i in range(len(idx) - 1):
        batches.append(np.array(inputs[ idx[i]: idx[i + 1]]))

    
    return np.array(batches)
    
def load_data(filename, train_split = 0.8, batch_size=50):
    #tr_loader, va_loader, te_loader, _ = get_train_val_test_loaders(
    #   num_classes=config('challenge.num_classes'))
    df = pd.read_csv(filename)
    x = df.filter(regex='obs*').values
    actions = np.array([[a] for a in df['f_actions'].values])
    
    x_a = np.append(x, actions, axis=1)
    x_a = compute_batches(x_a, batch_size)

    # Sigmoid of reward
    y =  df['reward'].values #.astype(int) #np.log(df['reward'].values + 1) / 10
    #one_hot = np.zeros((len(y), 2)) # 0 -->[1, 0], 1:[0, 1]
    #one_hot[np.arange(len(y)), y] = 1
    #y = one_hot
    y = compute_batches(y, batch_size)

    
    
    all_data = []
    for i in range(len(x_a)):
        all_data.append([torch.tensor(x_a[i]),
                                       torch.tensor(y[i]).long()])
    #all_data = np.array(all_data)
    
    
    training = all_data[:int(len(all_data) * train_split)]
    np.random.shuffle(training)
    #training = torch.tensor(training)

    
    validation = all_data[-int(len(all_data) * (1 - train_split)):]
    np.random.shuffle(validation)
    #validation = torch.tensor(validation)

    #import pdb; pdb.set_trace()
    
    return training, validation

def plot_training_stats(stats):
    plt.cla()
    stats_idx = stats.index.values# index is epoch
    for col in stats.columns:
        plt.plot(stats_idx, stats[col], label=col)

    plt.legend()
    plt.pause(0.00001)
    
def train_model():
    start_time = time()
    
    model = NN(fc_depth=1, hidden_units = 128,
                           input_dim=5, out_dim=2)
    model.double()
    loss_f = torch.nn.CrossEntropyLoss()
    #loss_f = torch.nn.L1Loss() # Mean Abs Error

    lr = 0.00001
    optimizer = torch.optim.Adam(model.parameters(), lr)
    
    '''
    checkpoint@(model, filename)
    '''
    start_epoch = 0
    num_epochs = 1
    batch_size = 100
    
    stats= pd.DataFrame(columns=['val_acc', 'val_loss', 'train_acc', 'train_loss'])
    training, validation = load_data(filename, train_split = 0.8, batch_size=batch_size)

    # Evaluate model with random initialization
    evaluate_epoch(training, validation, model, loss_f, start_epoch, stats)
    
    
    # Evaluate training
    for epoch in range(start_epoch, num_epochs):
        
        train_epoch(training, model, loss_f, optimizer)
        
        evaluate_epoch(training, validation, model, loss_f, epoch+1, stats)

        print(stats.tail())
        #plot_training_stats(stats)
        
        # save(model, epoch+1, params, stats)

    print('Finished Training')
    print(f'Completed in {round((time() - start_time)/60, 2)} minutes')
    print(stats)
    # Persist plot
    plt.ioff()
    plt.show()
    
    ##############
    # next: evaluate policy state_action in env
    return model, optimizer, loss_f

def eval_performance(agent, epochs, frames=250, viz=False):

    rewards = []
    for ep in range(epochs):
        obs = [env.reset()]
        total_reward = 0
        terminal = False
        ep_acts = []

        for f in range(frames):
            if viz and ep % (1 + (epochs//5)) == 0:
                env.render()
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
            
        if (1 + ep) % (1 + (epochs//5)) == 0:
            print("Terminal ep:", ep,
                      "\tep_rwd:", total_reward,)
                      #f"\tlr:{round(agent.lr, 2)}",
                      #"\tP(r_act):", round(agent.rand_act_prob, 2))
        
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

    #Plot observation axis for Q table
    '''
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
    '''
    return df

class NN_Agent_sa():
    def __init__(self, model, optimizer, loss_f):
        self.model_sa = model
        self.optimizer = optimizer
        self.loss_f =  loss_f
        self.outs = []

        self.timesteps = []
        self.act_preds = []
        self.choices = []

    def act(self, obs, time_t=0):
        s_a0 = torch.tensor(np.append(obs, 0))
        s_a1 = torch.tensor(np.append(obs, 1))
        out_a0 = np.array(model(s_a0).data)
        out_a1 = np.array(model(s_a1).data)
        self.act_preds.append(np.array([out_a0, out_a1]))
        #print(out_a0, out_a1)
        self.timesteps.append(time_t)

        #print(np.array(self.act_preds[-1]))
        expected_rew = np.array(self.act_preds[-1]) * np.array([-1, 1])
        expected_rew = np.sum(expected_rew, axis=1)
        #choice = np.argmin(np.array(self.act_preds[-1])[:, 0])
        choice = np.argmax(expected_rew)
        self.choices.append(choice)

        
        return choice #np.random.randint(2)
    
    def update_policy(self, action, state, next_state, reward, ep_acts=[]):
        train_xy = np.array([np.append(next_state, action), reward])
        print(train_xy)
        train_epoch(train_xy, self.model_sa, self.loss_f, self.optimizer)
        

    def plot_act_eval(self):
        self.timesteps = np.array(self.timesteps)
        #print(self.act_preds)
        #self.act_preds = np.array(self.act_preds)[:, 1]
        #self.act_preds = np.argmax(np.array(self.act_preds)[:, 1], axis = 1)
        self.act_preds =  np.array(self.act_preds) * np.array([-1, 1])
        self.act_preds = np.sum(self.act_preds, axis=2)
        #a0, a1self.act_preds[:, 0]
        
        #import pdb; pdb.set_trace()
        #print('a0\n', act_preds[:,0], 'a1\n', act_preds[:,1])
        df = pd.DataFrame({'timesteps':self.timesteps,
                                              'action0':self.act_preds[:, 0],
                                              'action1':self.act_preds[:, 1],
                                              'choice':np.array(self.choices)})
        df = df.groupby('timesteps').mean()
        
        print(df)
        df.plot()
        
        #timesteps = np.arange(len(a0))
        #plt.plot(timesteps, np.argmax(a0, axis=1), label="action zero")
        #plt.plot(timesteps, np.argmax(a1, axis=1), label="action one")
        #plt.plot(timesteps, :,1], label="action zero")
        #plt.plot(timesteps, np.array(a1)[:,1], label="action one")
        plt.legend()
        plt.show()

if __name__ == "__main__":
    filename = "random_ep_observation_rewards.csv"
    #save_observations(filename, 10000)
    observe_data(filename)
    
    model, optimizer, loss_f = train_model()

    agent = NN_Agent_sa(model, optimizer, loss_f)
    
    eval_performance(agent, 100, frames=250, viz=True  )
    agent.plot_act_eval()
    
    #plot_observation_reward(filename)
    
    #plt.show()
