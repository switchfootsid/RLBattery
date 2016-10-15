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
	day_chunk = 2
	total_years = 1
	episode_number = 0
	E_cap = 6.0
	P_cap = 3.0
	E_init = 0.3*E_cap
	epsilon = 0.5
	actions = np.arange(-P_cap, P_cap + 0.01, 0.5).tolist()
	actions.sort()
	total_number_hours = 24
	look_ahead = 1

	savings = list()

	#Creation of objects
	with open('model_store_k1_summer') as fp:
		models = pickle.load(fp)
	#starting main episode loop
	total_iterations = 1#total_years * day_chunk #day_chunk*total_years
	for i in range(50):
		grid_list = []
		reward_list = list()
		action_list = list()
		energy_list = list()
		price_list = list()
		load_action_list = list()
		episode_number = 0
		environment = Environment(gamma, eta, day_chunk, total_years)
		environment.setCurrentState(episode_number, E_init)
		learningAgent = LearningAgent(environment.currentState, actions, E_cap, P_cap, epsilon)
		funtionApproximator = FunctionApproximation('sgd', actions)
		funtionApproximator.models = models
		agent_bill, base_bill = 0.0, 0.0

		
		while(episode_number < total_iterations) :
			grid_list = list()
			cost_list = list()
			load_list = list()
			netload_list = list()
			solar_list = list()
			add = False
			for time in range(total_number_hours) :
				
				energy_list.append(learningAgent.currentState[1])

				action_sequence, rewardCumulative = learningAgent.getAction(episode_number, environment.currentState, funtionApproximator, environment, look_ahead, gamma, time)
				if action_sequence[0]== None:
					add = True
					break
				print "action action_sequence:",action_sequence			
				copy_current = deepcopy(environment.currentState)
				nextState, qvalue, isValid = environment.nextStep(episode_number, time, [learningAgent.actions[action_index] for action_index in action_sequence], look_ahead, funtionApproximator, learningAgent)
				action_taken = learningAgent.actions[action_sequence[0]]
				print action_taken

				#print('Episode', episode_number, 'current', learningAgent.currentState, 'action', action_taken,'next', nextState)
				grid_list.append(environment.getP_grid(copy_current, action_taken))
				action_list.append(action_taken)
				netload_list.append(learningAgent.currentState[0])
				load_list.append(environment.getLoad(episode_number, time))
				solar_list.append(environment.getSolar(episode_number, time))
				price_list.append(learningAgent.currentState[2]*100-4)
				cost_list.append(learningAgent.currentState[2])

				learningAgent.currentState = nextState

			episode_number += 1
			reward_list.append(rewardCumulative)
	 		agent_bill = sum([a*b for a,b in zip(cost_list, grid_list)])
			base_bill = sum([max(0, a*b) for a,b in zip(cost_list, netload_list)])
			if not add :
				savings.append(base_bill - agent_bill)
	
	print savings, len(savings)
	print 'Mean Savings:', np.mean(savings), '+/-', np.std(savings)
	#sns.distplot(savings)
	plt.plot(savings)
	plt.show()
	plt.plot(grid_list, label='Grid', c='r')
	plt.plot(action_list, label='Charge/Discharge', c='c')
	plt.plot(load_list, label='Load')
	plt.plot(price_list, label='Price (Scaled up)')
	plt.plot(solar_list, label='Solar Power', c='y')
	plt.legend(loc=1, mode="expand", borderaxespad=0.)
	plt.xticks(np.arange(1, len(grid_list)+1, 1.0))
	plt.yticks(np.arange(-3,11, 0.5))
	plt.xlabel('Hours')
	plt.ylabel('Grid')
	plt.show()
	
if __name__ == '__main__' :
    	main()