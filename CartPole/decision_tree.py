# Use a Decision Tree Classifier as the Policy in CartPole
# Developed by Nathan Shepherd

import gym
env = gym.make('CartPole-v0')

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
    for frame in range(ep_length):
      env.render()
      action = agent.act(observation)
      observation, reward, done, info = env.step(action)

      if done:
        print("Episode completed after %s timesteps" % str(frame))
        agent.remember(observation)
        break

class DecisionTree: 
  def __init__(self):
    # Each decision yeild a truth value after comparison
    # Each truth value is applied to the actions
    # The action with majority Truth value is chosen
    self.decisions = [0, 0, 0, 0]
    
    # Actions is an array representation of binary tree
    # Each leaf is the resulting action after a sequence of decisions
    self.actions  =  [-1 for i in range(4 * 4 + 1)]

  def get_leaf_index(self, obs):
    action_idx = 0
    for i in range(len(obs)):
      # left is false, right is true
      if obs[i] >= self.decisions[i]:
        action_idx = action_idx * 2 + 1
      else:
        action_idx = action_idx * 2 + 2

    return action_idx - 14 # num of parent nodes is 14

  def act(self, obs):
    idx = self.get_leaf_index(obs)

    if self.actions[idx] == -1:
      self.actions[idx] = env.action_space.sample()

    return self.actions[idx]

  def remember(self, obs):
    # Obs and previous action are taken to account
    # Called when episode reaches terminal state
    # TODO: change action only if there is less than average score
    prev = self.get_leaf_index(obs)
    if self.actions[prev] == 1:
      self.actions[prev] = 0
    else:
      self.actions[prev] = 1
    

if __name__ == "__main__":
  agent = DecisionTree()
  play_episode(10, agent)
  print(agent.actions)
