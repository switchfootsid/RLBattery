from __future__ import division
import numpy as np
import pandas as pd
import random 
from environment import Environment
from learningAgent import LearningAgent
from function_approximation import FunctionApproximation
from copy import deepcopy
import matplotlib.pyplot as plt
import seaborn as sns
import cPickle as pickle
import warnings
warnings.filterwarnings('ignore')

'''
To-Do:

- Diagnostic Class: Write a plotting routine/class for accumulated rewards/bill
	+ 
	+
	+ 
- Assess learning: What is the behaviour of the optimal policy? Do the rewards converge? 
- Initilization: Is learning sensitive to initialization/training of SVC model? 
- Function Approx: Batch-size, update and models (Random Forests, Kernel Regression etc)
- RL hyperparameters: Exploration annealing effect, gamma etc. 

'''

def main():
	isTrainingOn = True 
	gamma = 0.99 
	eta = 0.9 
	day_chunk = 10
	total_years = 2000
	episode_number = 0
	E_cap = 6.0
	P_cap = 3.0
	E_init = 0.3*E_cap
	epsilon = 1.0
	actions = np.arange(- P_cap, P_cap + 0.01, 0.5).tolist()
	#actions.sort()
	total_number_hours = 24
	look_ahead = 1

	batch = []
	
	miniBatchSize = 400
	bufferLength = 500
	lasting_list = []
	grid_list = []
	reward_list = []
	action_list = []
	
	#Creation of objects
	environment = Environment(gamma, eta, day_chunk, total_years)
	environment.setCurrentState(episode_number, E_init)
	learningAgent = LearningAgent(environment.currentState, actions, E_cap, P_cap, epsilon)
	funtionApproximator = FunctionApproximation('sgd', actions)

	#starting main episode loop
	total_iterations = total_years * day_chunk #day_chunk*total_years

	while(episode_number < total_iterations) :
		lasting = 1
		#print (episode_number)
		for time in range(total_number_hours) :
			
			'''
			Change in for loop by Siddharth: 
			1. Added exploration function in learningAgent.py
			2. Corrected isValid condition
			3. lasting_list : contains the time_steps upto which the agent reaches until failure (P_grid < 0)
			4. Edited nextStep in environment (currentState update)
			'''

			K = look_ahead
			
			if np.random.random() <= learningAgent.epsilon: 
				currentStateBackup = deepcopy(learningAgent.currentState)
				K = 0
				action_sequence, rewardCumulative = learningAgent.exploration(episode_number, time, environment, K)
			else:
				currentStateBackup = deepcopy(learningAgent.currentState)
				action_sequence, rewardCumulative = learningAgent.getAction(episode_number, learningAgent.currentState, funtionApproximator, environment, look_ahead, gamma, time)
				if action_sequence[0] == None: 
					break
					#print 'none actions', [actions[i] for i in learningAgent.getLegalActions(currentStateBackup)]
					#print 'none', currentStateBackup

			if action_sequence[0] == None:
				qvalue = -10
				funtionApproximator.update_qfunction(currentStateBackup, action_index, qvalue)



			nextState, qvalue, isValid = environment.nextStep(episode_number, time, [learningAgent.actions[action_index] for action_index in action_sequence], K, funtionApproximator, learningAgent)
			#print 'done'
			#print currentStateBackup, actions[action_sequence[0]], nextState

			action_index = action_sequence[0] 
			action_taken = learningAgent.actions[action_sequence[0]]
			action_list.append(action_taken)
			grid_list.append(environment.getP_grid(currentStateBackup, action_taken))
			#Experience Tuple
			###
			####currentStateBackup.append(action_taken) #indexed the actions to change experience tuple
			####batch.append((currentStateBackup, qvalue))
			'''
			currentStateBackup.append(qvalue)
			currentStateBackup.append(action_index)
			batch.append(currentStateBackup)

			if(len(batch) >= bufferLength) :
				miniBatch = random.sample(batch, miniBatchSize)
				funtionApproximator.update_qfunction(miniBatch, learningAgent)
				batch = []
			'''
			funtionApproximator.update_qfunction(currentStateBackup, action_index, qvalue)

			if (isValid) :
				episode_number += 1
				temp = environment.setCurrentState(episode_number, E_init)
				learningAgent.currentState = temp
				break

			learningAgent.currentState = nextState
			lasting += 1
			lasting_list.append(lasting)

		if (learningAgent.epsilon >= 0.0):
			learningAgent.epsilon -= 1/total_iterations
		
		if(episode_number%100 == 0) :
			print ("done with episode number = " + str(episode_number))
			print ("lasted days = ", len([1 for x in lasting_list if x >= 24]))
		
		episode_number += 1
		reward_list.append(rewardCumulative)
		
	plt.plot(action_list)
	plt.xlabel('Action value')
	plt.ylabel('Training Episodes')
	plt.show()
	plt.plot(grid_list)
	#plt.plot(reward_list)
	plt.xlabel('Grid value')
	plt.ylabel('Training Episodes')
	plt.show()
	print learningAgent.epsilon

	with open('model_store_k1_summer','w') as fp:
		pickle.dump(funtionApproximator.models, fp)

if __name__ == '__main__' :
    	main()