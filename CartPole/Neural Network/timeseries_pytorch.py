
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
    y =  df['reward'].values #np.log(df['reward'].values + 1) / 10
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
    
def main():
    start_time = time()
    
    model = NN(fc_depth=1, hidden_units = 16,
                           input_dim=5, out_dim=2)
    model.double()
    loss_f = torch.nn.CrossEntropyLoss()
    #loss_f = torch.nn.MSELoss()

    lr = 0.8 #0.00001
    optimizer = torch.optim.Adam(model.parameters(), lr)
    
    '''
    checkpoint@(model, filename)
    '''
    start_epoch = 0
    num_epochs = 3
    batch_size = 1000
    
    stats= pd.DataFrame(columns=['val_acc', 'val_loss', 'train_acc', 'train_loss'])
    training, validation = load_data(filename, train_split = 0.8, batch_size=batch_size)

    # Evaluate model with random initialization
    evaluate_epoch(training, validation, model, loss_f, start_epoch, stats)
    
    
    # Evaluate training
    for epoch in range(start_epoch, num_epochs):
        
        train_epoch(training, model, loss_f, optimizer)
        
        evaluate_epoch(training, validation, model, loss_f, epoch+1, stats)

        #import pdb; pdb.set_trace()
        plot_training_stats(stats)
        
        # save(model, epoch+1, params, stats)

    print('Finished Training')
    print(f'Completed in {round((time() - start_time)/60, 2)} minutes')
    print(stats)
    # Persist plot
    plt.ioff()
    plt.show()
    
    ##############
    # TODO: evaluate policy state_action in env
    
if __name__ == "__main__":
    filename = "random_ep_observation_rewards.csv"
    #save_observations(filename, 10000)
    observe_data(filename)
    
    main()
    
    #plot_observation_reward(filename)
    
    #plt.show()
