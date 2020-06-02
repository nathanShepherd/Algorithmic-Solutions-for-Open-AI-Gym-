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

def play_episode(epochs, agent, ep_length=200):
  for episode in range(epochs):
    observation = env.reset()
    ep_reward = 0
    ep_obs = []
    
    for frame in range(ep_length):
      env.render()
      action = agent.act(observation)
      observation, reward, done, info = env.step(action)
      
      ep_obs.append(observation)
      ep_reward += reward
      
      if done:
        print("Episode completed after %s timesteps" % str(frame))
        agent.terminal(ep_obs, ep_reward)
        break

class MarkovChain: 
  def __init__(self):
    self.decisions = [0, 0, 0, 0]

    self.observation_mem = []

  def fit_transition_prob(self, obs_by_time, n, m):
    # obs_by_time = states at time t in all obs
    # current_states = rows in table = n
    # next states = columns of table = m
    
    self.t_prob  = np.zeros((n, m))
    current_states_count = np.zeros((n))

    # t_prob <-- P( current_state and next_state)
    for i, obs_t in enumerate(obs_by_time[:-1]):
      self.t_prob[obs_t][obs_by_time[i + 1]] += 1
      current_states_count[obs_t]  += 1

    # t_prob <-- P( current_state and next_state) / P(current_state)
    for state_idx in range(len(current_states_count)):
      if state_idx == 0:
        continue
      self.t_prob[state_idx] /= current_states_count[state_idx]

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
    if training:
      # Only remember cart position
      obs = obs[2]
      self.observation_mem.append(obs)
      
    return np.random.randint(0, 2)

  def terminal(self, obs, reward):
    print(f'{Terminal obs: {obs}, reward: {reward}')
    self.observation_mem.append(obs)

    self.observation_mem = list(lambda obs: round(obs, 0),
                                                      self.observation_mem)
    
    agent.fit_transition_prob(self.observation_mem,
                                                  max(self.observation_mem))
    

def test_MM():
  agent = MarkovChain()

  obs_by_time = [1, 2, 1, 0]
  num_states = 3

  agent.fit_transition_prob(obs_by_time, num_states)
  
  print("Observations", obs)
  print("Transition Table:\n", agent.t_prob)
  
  for state in range(num_states):
    E_i = agent.expected_consecutive_obs(state)
    print(f'State: {state}, E[observing {state}]: {E_i}')

def main():  
  agent = MarkovChain()
  play_episode(10, agent)
  print(agent.actions)

if __name__ == "__main__":
  main() #test_MM()
