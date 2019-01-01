# Use a Decision Tree Classifier as the Policy in CartPole
# Developed by Nathan Shepherd

import gym
env = gym.make('CartPole-v0')

actions = []# 1 or 0
rewards = []# 1 at every timestep
'''
observation                 Min         Max
  0	Cart Position             -4.8            4.8
  1	Cart Velocity             -Inf            Inf
  2	Pole Angle                 -24°           24°
  3	Pole Velocity At Tip      -Inf            Inf
'''
obs = []
for episode in range(10):
  observation = env.reset()

  for frame in range(100):
    env.render()
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)

    actions.append(action)
    rewards.append(reward)
    obs.append(observation)
    if done:
      print("Episode completed after %s timesteps" % str(frame))
      break

for o in obs:
  print(o)
    

