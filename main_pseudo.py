
import numpy as np
import random

BUFFER_LEN = 100
MINI_BATCH_SIZE = 50
batch = []
reward_plots = []



for episode in range(max_episode):

	currentState = env_instance.get_initial_state(episode) #Initial state for the day of the training data. 
	
	for time in range(T):
		
		# 1. Based on 2-step look-ahead, choose an action.
		
		action = agent_instance.choose_action(currentState) 

		# 2. Execute the action in simulation
		
		nextState, reward, condition, finalState = env_instance.next_step(currentState, time)

		# 3. Grow the experience batch with (S,A,R,S1) and perform conditional checking.
		# NOTE - Perform conditional checking on reward in the Function Approx module.
		
		experience = (currentState, kQmax) # (currentState, Q)

		if len(batch) > BUFFER_LEN:
			#4. Refresh the batch and update the Q-function.
		
			minibatch = random.sample(batch, MINI_BATCH_SIZE)
			function_approx_instance.update_qfunction(minibatch, agent_instance)
			batch = [] #refresh batch
		
		else:
		
			batch.append(experience)

		if condition == False:
			reward_plots.append(reward)
			break

		currentState = nextState
		reward_plots.append(reward)

	#Annealing
	agent_instance.epsilon -= 1/()
	agent_instance.alpha -= 






