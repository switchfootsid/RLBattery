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

def main():
	gamma = 0.89 
	eta = 0.9 
	day_chunk = 1 
	total_years = 1 
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
	
	miniBatchSize = 50
	bufferLength = 100
	lasting_list = []
	grid_list = []
	reward_list = list()
	action_list = list()
	energy_list = list()
	grid_list = list()
	load_list = list()
	load_action_list = list()

	#Creation of objects
	with open('model_store') as fp:
		model = pickle.load(fp)
	print model

	environment = Environment(gamma, eta, day_chunk, total_years)
	environment.setCurrentState(episode_number, E_init)
	learningAgent = LearningAgent(environment.currentState, actions, E_cap, P_cap, epsilon)
	funtionApproximator = FunctionApproximation('svr')
	funtionApproximator.model = model
	
	#starting main episode loop
	total_iterations = total_years * day_chunk #day_chunk*total_years

	while(episode_number < total_iterations) :
		
		for time in range(total_number_hours) :
			
			energy_list.append(learningAgent.currentState[1])

			action_sequence, rewardCumulative = learningAgent.getAction(episode_number, environment.currentState, funtionApproximator, environment, look_ahead, gamma, time)
			copy_current = deepcopy(environment.currentState)
			nextState, qvalue, isValid = environment.nextStep(episode_number, time, [learningAgent.actions[action_index] for action_index in action_sequence], look_ahead, funtionApproximator, learningAgent)
			action_taken = learningAgent.actions[action_sequence[0]]
			print action_taken


			#print('Episode', episode_number, 'current', learningAgent.currentState, 'action', action_taken,'next', nextState)
			grid_list.append(environment.getP_grid(copy_current, action_taken))
			action_list.append(action_taken)
			load_list.append(learningAgent.currentState[0])
			load_action_list.append(action_taken+learningAgent.currentState[0])

			learningAgent.currentState = nextState

		if(episode_number%100 == 0) :
			print ("done with episode number = " + str(episode_number))
			#print ("lasted days = ", len([1 for x in lasting_list if x >= 24]))
		
		episode_number += 1
		reward_list.append(rewardCumulative)
		
	plt.plot(grid_list, label='Grid')
	#plt.plot(energy_list, label='Energy Level')
	plt.plot(load_list, label='Load')
	plt.plot(action_list, label='Actions')
	plt.legend(loc=1, mode="expand", borderaxespad=0.)

	plt.xlabel('Hours')
	plt.ylabel('Grid')
	plt.show()
	
if __name__ == '__main__' :
    	main()