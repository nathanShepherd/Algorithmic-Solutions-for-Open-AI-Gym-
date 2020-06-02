import gym
import numpy as np
env = gym.make('CartPole-v0')

# Use a Decision Tree Classifier as the Policy in CartPole
# Developed by Nathan Shepherd


actions = []# 1 or 0
obs = []# 4 by N vector
rewards = []# 1 at every timestep
'''
observation                 Min         Max
  0	Cart Position             -4.8            4.8
  1	Cart Velocity             -Inf            Inf
  2	Pole Angle                 -24           24
  3	Pole Velocity At Tip      -Inf            Inf
'''

def play_episode(epochs, agent, ep_length=200, viz=True):
  for episode in range(epochs):
    observation = env.reset()
    ep_reward = 0
    f_reward = []
    ep_obs = []
    
    for frame in range(ep_length):
      if viz:
        env.render()
      action = agent.act(observation)
      observation, reward, done, info = env.step(action)
      
      ep_obs.append(observation)
      ep_reward += reward

      if done:
        print("\nEpisode", episode," completed after %s timesteps" % str(frame))
        f_reward.append(0)
        agent.terminal(ep_obs, f_reward)
        break
      
      f_reward.append(ep_reward)
      
      
      

class MarkovChain: 
  def __init__(self):
    self.decisions = [0, 0, 0, 0]

    self.observation_mem = []
    self.t_prob = None

  def fit_transition_prob(self, obs_by_time, n_states):
    # obs_by_time = states at time t in all obs
    # current_states = rows in table = n
    # next states = columns of table = m

    #import pdb; pdb.set_trace()

    local_table = np.zeros((n_states, n_states))
    
    if type(self.t_prob) == type(None):
      self.t_prob  = np.zeros((n_states, n_states))
    else:
      # Record current probs to update later
      local_table = self.t_prob
      
      # pad smaller transition table with zeros
      if n_states > len(self.t_prob[0]):
        pad_length = n_states - len(self.t_prob[0])
        self.t_prob = np.pad(self.t_prob, ((0, pad_length),( 0, pad_length)),
                                             constant_values=(0))
        local_table = self.t_prob
        
      elif  n_states < len(self.t_prob[0]):
        n_states = len(self.t_prob[0])
        
        
    current_states_count = np.zeros((n_states))

    # t_prob <-- P( current_state and next_state)
    for i, obs_t in enumerate(obs_by_time[:-1]):
      self.t_prob[obs_t][obs_by_time[i + 1]] += 1
      current_states_count[obs_t]  += 1

    # t_prob <-- P (s_(t + 1) | s_t) = P( current_state and next_state) / P(current_state)
    for state_idx in range(len(current_states_count)):
      if current_states_count[state_idx] != 0:
        self.t_prob[state_idx][:] /= current_states_count[state_idx]

    # Take the mean prob of prev estimate for t_prob and current prob
    self.t_prob = np.mean((self.t_prob.flatten(),
                                             local_table.flatten()), axis = 0)
    
    self.t_prob = np.reshape(self.t_prob, (n_states, n_states))

    # Normalize mean probabilites to one
    for i in range(len(self.t_prob)):
      ssum =  sum(self.t_prob[i])
      
      if ssum != 0:
        self.t_prob[i] = self.t_prob[i] /ssum
      else:
        self.t_prob[i] = 0
      #print( self.t_prob[i], 'sum: ',   sum(self.t_prob[i]))
      
  def prob_states_given_model(self, states):
    out_prob = 1
    for obs_t in range(len(states) - 1):
      out_prob *= self.t_prob[ states[obs_t] ][ states[obs_t + 1] ]
    return out_prob

  def expected_consecutive_obs(self, state):
    return 1 / (1 - self.t_prob[state][state])
    

  def proj_next_state_obs(self, current_obs):
    pass
    
  def act(self, obs, training=True):
    '''
    if training:
      # Only remember cart position
      
      obs = int(obs[2] * 10)
      self.observation_mem.append(obs)
    '''
    return np.random.randint(0, 2)

  def sigmoid_array(self, X):
    return 1 / (1 + np.exp( -X))
  
  def terminal(self, obs, reward):
    # Only observe cart position
    cleaned_obs = self.sigmoid_array(np.array(obs)[:,2])
    
    print('{Terminal obs: {', cleaned_obs, '}, reward: {', reward, '}')
      
    #self.act(obs)
    
    self.observation_mem = list(map(lambda x: int( round(x, 0)), cleaned_obs))
    print("Model's Observation_Memory: \n", self.observation_mem)
    
    self.fit_transition_prob(self.observation_mem,
                                                  max(self.observation_mem) + 1)

    #self.view_model_params()

  def view_model_params(self):
    print("Model Parameters:")
    print("=============")
    print("Transition Table:")
    for i, row in enumerate(self.t_prob):
      if i == 0:  
        pass #print(len(self.t_prob[row]))
      row = list(map(lambda x: round(x, 3), row))
      print("P( curr_obs = ", i, " | next_obs  = col )", row)
      
    for state in range(len(self.t_prob)):
      E_i = self.expected_consecutive_obs(state)
      print('State: {', state, '}, E[observing {state}]: {', E_i, '}')

def test_MM():
  agent = MarkovChain()

  obs_by_time = [1, 1, 1, 0]
  num_states = 2

  #import pdb; pdb.set_trace()
  
  agent.fit_transition_prob(obs_by_time, num_states)
  
  print("Observations", obs_by_time)
  print("Transition Table:\n", agent.t_prob)
  
  for state in range(num_states):
    E_i = agent.expected_consecutive_obs(state)
    print('State: {', state, '}, E[observing {state}]: {', E_i, '}')

'''
Agent combines multiple fully observed Markov Chains
'''
class MarkovAgent():
  def __init__(self):
    self.obs_to_obs = MarkovChain()
    self.act_obs = MarkovChain()
    self.obs_act = MarkovChain()
    self.act_reward = MarkovChain()
    
    self.actions = []
    self.rewards = np.array([])
    self.training = True

  def act(self, obs):
    if self.training:
      action = np.random.randint(0, 2)
      self.actions.append(action)
      return action
    else:
      return np.random.randint(0, 2)
      '''
      if type(self.obs_act.t_prob) == type(None):
        return np.random.randint(0, 2)
      else:
        # TODO: Determine action with greatest reward
        cleaned_obs = self.sigmoid_array(np.array([obs[2]]))[0]
        cleaned_obs = int( round(cleaned_obs, 0))
        next_state = np.argmax(self.obs_to_obs.t_prob[cleaned_obs])
        act_taken = np.argmax(self.obs_act.t_prob[next_state])
        #act_taken = np.argmax(self.act_reward.t_prob[:,-1])
        return act_taken
      '''
      
  def sigmoid_array(self, X):
    num_bins = 1
    # 10 bin = (z-100, ..., z0, ... z100 --> (01, ..., x5, x6, x7))
    # which is equal to 7 action bins in the range(-100, 100)
    # where z10 == 10, x5 == 0.5
    X = num_bins / (1 + np.exp( -0.1*X))
    return list(map(lambda x: int( round(x, 0)), X))
    
  
    
  def terminal(self, obs, reward):
    '''
observation                 Min         Max
  0	Cart Position             -4.8            4.8
  1	Cart Velocity             -Inf            Inf
  2	Pole Angle                 -24           24
  3	Pole Velocity At Tip      -Inf            Inf
    '''
    cleaned_obs = self.sigmoid_array(np.array(obs)[:,0])
    
    #print('{Terminal obs: {', cleaned_obs, '}, Terminal reward: {', reward, '}')
    self.obs_to_obs.observation_mem = cleaned_obs
    #print("Obs: ", self.obs_to_obs.observation_mem)
    
    self.obs_to_obs.fit_transition_prob(self.obs_to_obs.observation_mem,
                                                  max(self.obs_to_obs.observation_mem) + 1)

    reward = list(map(lambda x: int( round(np.log((x + 1) *10), 0)), reward))
    #print('Reward: ', self.rewards, reward)

    self.rewards = np.concatenate((self.rewards, reward))
    #print("Act:  ", self.actions)
    #print('Reward: ', self.rewards)
    
    
    obs_by_actions_reward = np.array(list(zip(self.obs_to_obs.observation_mem,
                                                                                 self.actions, reward))).astype(int)
    #print('obs_by_actions_reward', obs_by_actions_reward)
    for col in range(len(obs_by_actions_reward[0]) -1 ):
      if col == 0:
        part = obs_by_actions_reward[:, col:col + 2]
        #part_reverse = np.flip(obs_by_actions_reward[:, col:col + 2], axis =1)
        #print('act/obs: ', part) ###
        for row in part:
          upper_limit = max(row) + 1        
          self.act_obs.fit_transition_prob(row, upper_limit)

        # Should not be the same?
        #for row in part_reverse:
        #  self.obs_act.fit_transition_prob(np.flip(row), upper_limit)
          
      elif col == 1:
        part = obs_by_actions_reward[:, col:col+2]
        #print("act/rew", part) ###
        for row in part:
          upper_limit = max(row) + 1
          self.act_reward.fit_transition_prob(row, upper_limit)
        

    #self.actions = []
    #self.training = False

  def view_model_params(self):
    print("\n row, col  --> Act,  Reward")
    self.act_reward.view_model_params()

    print("\nrow = col --> Obs = Obs")
    self.obs_to_obs.view_model_params()

    print("\nrow = col --> Act = Obs")
    self.act_obs.view_model_params()
    
    print("\nrow = col --> Obs = Act")
    self.act_obs.view_model_params()
    
def main():  
  agent = MarkovAgent() #MarkovChain()
  play_episode(300, agent, viz=False)
  agent.view_model_params()

if __name__ == "__main__":
  main()
  #test_MM()
