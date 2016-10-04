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
	isTrainingOn=True 
	gamma = 0.89 
	eta = 0.9 
	day_chunk = 15 
	total_years = 100
	episode_number = 0
	E_cap = 6.0
	P_cap = 3.0
	E_init = 0.3*E_cap
	epsilon = 0.5
	actions = np.arange(-P_cap, P_cap + 0.01, 0.50).tolist()
	actions.sort()
	total_number_hours = 24
	look_ahead = 0

	batch = []
	
	miniBatchSize = 200
	bufferLength = 500
	lasting_list = []
	grid_list = []
	reward_list = []
	action_list = []
	
	#Creation of objects
	environment = Environment(gamma, eta, day_chunk, total_years)
	environment.setCurrentState(episode_number, E_init)
	learningAgent = LearningAgent(environment.currentState, actions, E_cap, P_cap, epsilon)
	funtionApproximator = FunctionApproximation('sgd')

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
				#print 'Acting randomly', currentStateBackup
				K = 0
				action_sequence, rewardCumulative = learningAgent.exploration(episode_number, time, environment, 0)
				#print 'random', rewardCumulative
			else:
				currentStateBackup = deepcopy(learningAgent.currentState)
				#print 'Acting thoughtfully', currentStateBackup
				action_sequence, rewardCumulative = learningAgent.getAction(episode_number, learningAgent.currentState, funtionApproximator, environment, look_ahead, gamma, time)
				#print 'thoughtful', rewardCumulative
				action_list.append(learningAgent.actions[action_sequence[0]])
			nextState, qvalue, isValid = environment.nextStep(episode_number, time, [learningAgent.actions[action_index] for action_index in action_sequence], K, funtionApproximator, learningAgent)
			action_taken = learningAgent.actions[action_sequence[0]]

			#print qvalue, action_taken
			#print('Episode', episode_number, 'current', currentStateBackup, 'action', action_taken,'next', nextState)
			
			#Experience Tuple
			currentStateBackup.append(action_taken) #indexed the actions to change experience tuple
			batch.append((currentStateBackup, qvalue))
			grid_list.append(environment.getP_grid(environment.currentState, action_sequence[0]))
			#action_list.append(action_taken)

			if(len(batch) >= bufferLength) :
				miniBatch = random.sample(batch, miniBatchSize)
				funtionApproximator.update_qfunction(miniBatch, learningAgent)
				batch = []
			
			if(isValid) :
				episode_number += 1
				temp = environment.setCurrentState(episode_number, E_init)
				learningAgent.currentState = temp
				break

			learningAgent.currentState = nextState
			lasting += 1

		if (learningAgent.epsilon >= 0.0):
			learningAgent.epsilon -= 1/total_iterations
		'''
		if(episode_number%500 == 0) :
			print ("done with episode number = " + str(episode_number))
			print ("lasted days = ", len([1 for x in lasting_list if x >= 24]))
		'''
		episode_number += 1
		lasting_list.append(lasting)
		#reward_list.append(rewardCumulative)
		
	plt.plot(action_list)
	plt.xlabel('Action value')
	plt.ylabel('Training Episodes')
	plt.show()
	with open('model_store','w') as fp:
		pickle.dump(funtionApproximator.model, fp)


if __name__ == '__main__' :
    	main()