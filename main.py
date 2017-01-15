from __future__ import division
import numpy as np
import pandas as pd
import random 
from environment import Environment
from learningAgent import LearningAgent
from function_approximation import FunctionApproximation
from collections import defaultdict
from copy import deepcopy
import matplotlib.pyplot as plt
import seaborn as sns
import cPickle as pickle
import warnings
warnings.filterwarnings('ignore')

def main(start_date, day_chunk, eta, E_cap, P_cap, epsilon, total_years, price_scheme, DOD, name):
	isTrainingOn = True 
	gamma = 0.89
	#eta = 0.9 
	#day_chunk = 20
	#total_years = 5
	episode_number = 0
	#E_cap = 6.4 
	#P_cap = 3.0 
	E_init = (1-DOD)*E_cap # 0.3*E_cap 
	#epsilon = 0.3
	actions = np.arange(-P_cap, P_cap + 0.01, 0.5).tolist()
	#actions.sort()
	total_number_hours = 24
	look_ahead = 2

	batch = []
	
	miniBatchSize = 200
	bufferLength = 100
	lasting_list = []
	grid_list = []
	reward_list = []
	action_list = []
	
	#Creation of objects
	environment = Environment(gamma, eta, total_years, start_date, day_chunk, price_scheme)
	environment.setCurrentState(episode_number, E_init)
	learningAgent = LearningAgent(environment.currentState, actions, E_cap, P_cap, epsilon)
	funtionApproximator = FunctionApproximation('extra_trees', actions)

	#Pickle the model
	"""
	with open('./models/fqi_winter.pkl', 'rb') as fp:
		models = pickle.load(fp)
	
	funtionApproximator.models = models
	"""

	with open('./featurizer.pkl', 'rb') as ff:
		featurizer = pickle.load(ff)
	
	funtionApproximator.featurizer = featurizer

	total_iterations = total_years * day_chunk #day_chunk*total_years
	lasting = 1

	batch = defaultdict(list)
	targets = defaultdict(list)
	
	while(episode_number < total_iterations) :
		
		explore = 'no'
		
		for time in range(total_number_hours) :

			K = look_ahead + 1
			
			if np.random.random() <= learningAgent.epsilon: 
				currentStateBackup = deepcopy(learningAgent.currentState)
				K = 1
				explore = 'yo'
				action_sequence, rewardCumulative = learningAgent.exploration(episode_number, time, environment, K, funtionApproximator, gamma)
				if None in action_sequence:
					print 'random', action_sequence
			else:
				currentStateBackup = deepcopy(learningAgent.currentState)
				action_sequence, rewardCumulative = learningAgent.getAction(episode_number, learningAgent.currentState, funtionApproximator, environment, K, gamma, time)
				if None in action_sequence:
					print 'not random', action_sequence

			if None in action_sequence:
				print currentStateBackup, action_sequence, rewardCumulative 
				explore = 'no'
				'''
				if action_sequence[0] == None:
					for action_index in learningAgent.getLegalActions(currentStateBackup):
						funtionApproximator.update_qfunction(currentStateBackup, action_index, rewardCumulative)
				else:	
					funtionApproximator.update_qfunction(currentStateBackup, action_sequence[0], rewardCumulative)
				break
				'''
				if action_sequence[0] == None:
					for action_index in learningAgent.getLegalActions(currentStateBackup):
						batch[action_index].append(currentStateBackup)
						targets[action_index].append(rewardCumulative)
				else:
					batch[action_sequence[0]].append(currentStateBackup)
					targets[action_sequence[0]].append(rewardCumulative)
				break

			nextState, qvalue, isValid = environment.nextStep(episode_number, time, [learningAgent.actions[action_index] for action_index in action_sequence], K, funtionApproximator, learningAgent)

			action_index = action_sequence[0] 
			action_taken = learningAgent.actions[action_sequence[0]]
			action_list.append(action_taken)
			grid_list.append(environment.getP_grid(currentStateBackup, action_taken))
			#Experience Tuple
			###
			####currentStateBackup.append(action_taken) #indexed the actions to change experience tuple
			####batch.append((currentStateBackup, qvalue))
			
			batch[action_index].append(currentStateBackup)
			targets[action_index].append(qvalue)
			
			if(episode_number % bufferLength == 0) :
				funtionApproximator.update_qfunction(batch, targets)
				batch = defaultdict(list)
				targets = defaultdict(list)
			'''
			funtionApproximator.update_qfunction(currentStateBackup, action_index, qvalue)
			'''
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
		
		if(episode_number%10 == 0) :
			print ("done with episode number = " + str(episode_number))
			print ("lasted days = ", len([1 for x in lasting_list if x >= 24]))
			
			file_name = './agent_' + name + '.pkl'  
			#with open('./models/fqi_winter.pkl', 'wb') as fp:
			with open(file_name, 'wb') as fp:
				pickle.dump(funtionApproximator.models, fp)
			
			print 'saving ...'
		episode_number += 1
		reward_list.append(rewardCumulative)
		
	#plt.plot(action_list)
	#plt.xlabel('Action value')
	#plt.ylabel('Training Episodes')
	#plt.show()
	
	#plt.plot(grid_list)
	#plt.ylabel('Grid value')
	#plt.xlabel('Training Episodes')
	#plt.show()
	#print learningAgent.epsilon

	#with open('./models/fqi_winter.pkl', 'wb') as fp:
	with open(file_name, 'wb') as fp:
		pickle.dump(funtionApproximator.models, fp)

	#with open('./models/fqi_winter.pkl','wb') as fp:
	#	pickle.dump(funtionApproximator.models, fp)
	
	with open('./featurizer.pkl', 'w') as ff:
		pickle.dump(funtionApproximator.featurizer, ff)
	
#if __name__ == '__main__' :
#    	main()
