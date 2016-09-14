import numpy as np
import pandas as pd
import random 
from __future__ import division

def main():
	gamma = 0.9
	eta = 0.9
	day_chunk = 15
    total_years = 500
  	episode_number = 0
  	E_cap = 6.0
  	P_cap = 3.3
  	E_init = 0.3*E_cap
  	epsilon = 0.5
  	actions = np.arange(-P_cap, P_cap, 0.5)
  	total_number_hours = 24
  	look_ahead = 2
  	batch = []
  	miniBatchSize = 50
  	bufferLength = 100
  	reward_plot = None #object for plotting class

  	#creation of objects
  	environment = Environment(gamma, eta, day_chunk, total_years)
  	environment.setCurrentState(episode_number, E_init)
  	learningAgent = LearningAgent(environment.currentState, actions, E_cap, P_cap, epsilon)
  	funtionApproximator = FunctionApproximation('svr')

  	#starting main episode loop
  	total_iterations = day_chunk*total_years
  	
  	while(episode_number < total_iterations) :
  		
  		for time in range(total_number_hours) :

  			action_sequence, rewardCumulative = learningAgent.getAction(learningAgent.currentState, funtionApproximator, environment, look_ahead, gamma)
  			currentStateBackup = environment.currentState
  			nextState, qvalue, isValid = environment.nextStep(action_sequence, look_ahead, funtionApproximator, learningAgent)
  			batch.append([currentStateBackup, action_sequence[0], qvalue])
  			
  			if(len(batch) >= bufferLength) :
  				miniBatch = random.sample(batch, miniBatchSize)
  				learningAgent.update_qfunction(miniBatch, learningAgent)
  				batch = []
  			learningAgent.currentState = nextState
  			
  			if(not isValid) :
  				#reward_plot.something
  				break

  		#Annealing of epsilon
  		learningAgent.epsilon -= 1/iterations  
  		 
  		if learningAgent.alpha <= 0.10: #Annealing of alpha
  			learningAgent.alpha *= 0.993
  		
  		environment.setCurrentState(episode_number, E_init)
