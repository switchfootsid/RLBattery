from __future__ import division
import numpy as np
import pandas as pd
import random 
from environment import Environment
from learningAgent import LearningAgent
from function_approximation import FunctionApproximation

'''
To-Do:
- Debug learning and convergence - hyperparameters 
- 
'''


def main():
	gamma = 0.9
	eta = 0.9
	day_chunk = 15
	total_years = 500
	episode_number = 0
	E_cap = 6.0
	P_cap = 3.0
	E_init = 0.3*E_cap
	epsilon = 0.5
	actions = np.arange(-P_cap, P_cap + 0.01, 0.5).tolist()+[0]
	actions.sort()
	total_number_hours = 24
	look_ahead = 2
	batch = []
	miniBatchSize = 5
	bufferLength = 10
	reward_plot = None #object for plotting class

	#creation of objects
	environment = Environment(gamma, eta, day_chunk, total_years)
	environment.setCurrentState(episode_number, E_init)
	learningAgent = LearningAgent(environment.currentState, actions, E_cap, P_cap, epsilon)
	funtionApproximator = FunctionApproximation('svr')

	#starting main episode loop
	total_iterations = 100#day_chunk*total_years

	while(episode_number < total_iterations) :

		print (episode_number)
		for time in range(total_number_hours) :
			action_sequence, rewardCumulative = learningAgent.getAction(episode_number, learningAgent.currentState, funtionApproximator, environment, look_ahead, gamma, time)

			print("action sequences",action_sequence,'cumilative reward',rewardCumulative)			
			currentStateBackup = environment.currentState
			#print("action sequences are fucked up",action_sequence)
			#print("this i guess is fucked up",[learningAgent.actions[action_index]for action_index in action_sequence])
			nextState, qvalue, isValid = environment.nextStep(episode_number, time, [learningAgent.actions[action_index]for action_index in action_sequence], look_ahead, funtionApproximator, learningAgent)
			print('nextState',nextState,'currentState',environment.currentState)
			currentStateBackup.append(action_sequence[0])
			batch.append((currentStateBackup, qvalue))

			if(len(batch) >= bufferLength) :
				miniBatch = random.sample(batch, miniBatchSize)
				funtionApproximator.update_qfunction(miniBatch, learningAgent)
				batch = []
			
			learningAgent.currentState = nextState
			if(not isValid) :
				break
			print("done....")
		#Annealing of epsilon
		episode_number += 1
	        learningAgent.epsilon -= 1/total_iterations  
	'''        
    	if learningAgent.alpha <= 0.10: #Annealing alpha
    		learningAgent.alpha *= 0.993
    	'''
    	environment.setCurrentState(episode_number, E_init)


if __name__ == '__main__' :
    	main()
