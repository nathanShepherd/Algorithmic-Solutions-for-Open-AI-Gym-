# Algorithmic Solutions for Open AI Gym

Using machine learning and computer science fundamentals to solve [Open AI Gym](https://gym.openai.com/) problems.

![reinforcement_learning](/graphs/reinforcement_learning.png)

policy: input_frame --> output_action

update policy on done:

	if success:
		increase probability of action
		sequence given observations
	else:
		decrease P(act | obs)

	cons:
		sample innefficiency
		sparse rewards

	solutions to cons:
		reward shaping 
			- manual and can be misleading
			- can result in local minimum
			- specifically a hard problem

		solve auxilary supervised taks: (DeepMind)
			- Most valuable feature(s) extraction 
			--> Maximize subset of features total change

			- Reward prediction (current state)
			--> Determine reward for an input 
			--> given previous sequence of input

			- Value function replay (total future states)
			--> Determine E[Reward] of all future states

		state dynamics (Markov Model):
			- E[next_state] given the current state and action
			- Utilize state exploration 
			- for states with uncertertain reward

		hindsight experience replay:
			- Learn from any epoch (even if unsuccessfull)
			--> The goal for each epoch:
			--> 	sequence of (s, a) that lead to final state
			--> Goals are stored in experience buffer
			--> Each episode tries to reach a specific goal

		model-free learning (Q-Learning):
			- Store combinations of states and actions in Q
			- Estimate discounted future reward at terminal
			- Update Q[state][action] with actual reward



