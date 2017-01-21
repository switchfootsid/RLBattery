from copy import deepcopy
import pandas as pd

debug = not True

class Environment :
	def __init__(self, Gamma, eta, total_years, start_date, day_chunk, price_scheme) :
		self.eta = eta #(battery efficiency)
		self.Gamma = Gamma
		self.start = start_date #pick season 
		self.day_chunk = day_chunk
		self.df_solar = pd.read_csv('./solar_double.csv')
	    	self.df_load = pd.read_csv('./load_data_peak6.csv') # %Peak Load
		self.training_time = total_years 
		self.diff = (self.df_load.ix[self.start:self.start+self.day_chunk-1] - self.df_solar.ix[self.start:self.start+self.day_chunk-1]) #change here by Siddharth, just take first 15 days 
    		#self.diff = (self.df_load - self.df_solar) 
    		self.net_load = pd.concat([self.diff]*self.training_time, ignore_index=True).values.tolist()
    		self.price_scheme = price_scheme
    		self.currentState = None
		
	def setCurrentState(self, episode_number, E_init):
		'''
		Set's the initialState (0th hour) for episode_number.
		episode_number 
		E_init - passed by agent class
		CHANGE HERE: returns Initial State (currentState in this case)
		'''
		net_load = float(self.net_load[episode_number][0])
		energy_level = E_init
		price = self.getPrice(0)
		
		#initialState = [net_load, energy_level, price] 
		initialState =  [net_load, energy_level, price, 0]
		self.currentState = initialState

		return initialState
		
	### CHANGE HERE --- 3rd October ---
	def nextStep(self, episode_number, time_step, action_sequence, k, FA, agent) :
		'''
		Perform constraint checking (energy, grid) and assign penalty/rewards.
	        Output: reward/penalty, next state, constraint satisfaction (boolean)

	        CHANGE HERE by Siddharth: 
	        currentStateBackup passed from main(), this ensures coherent state updates between learningAgent, main() and environment classes. 
	        PLEASE CHECK.
	        '''
	        lastState, cumulativeReward = self.getCumulativeReward(episode_number, time_step, k, action_sequence)
	        self.currentState, reward, isValid = self.getNextState(episode_number, time_step, self.currentState, action_sequence[0])

	        if lastState == None:
	        	# bad action P_grid < 0
	        	return self.currentState, cumulativeReward, isValid

	       	qValueLastState = FA.predictQvalue(lastState, agent.getLegalActions(lastState))

	        #if k == 0: ### CHANGE HERE
	        #	return self.currentState, cumulativeReward + self.Gamma * qValueLastState, isValid
		
		return self.currentState, cumulativeReward + (self.Gamma**(k))*qValueLastState, isValid

	
	def getCumulativeReward(self, episode_number, time_step, k, actions) :
		#print("actions in cumulativeReward",actions)
		gamma = 1
		state = deepcopy(self.currentState)
		cr = 0

		### CHANGE HERE -- 3rd October --
		for i in range(0, k) :
			#print("actions in get nextstate",actions)
			lastState, reward, isValid = self.getNextState(episode_number, time_step, state, actions[i])
			#print 'state', state, reward, isValid
			time_step += 1
			if (isValid) :
				return None, reward 
			cr += gamma*reward 
			gamma *= self.Gamma
			state = deepcopy(lastState)  ### CHANGE HERE -- 3rd October --

		return state, cr # target = (R1 + gamma*R2 + gamma2*Qmax)
	
	def getP_grid(self, state, action) :

		if action >= 0:
            		P_charge, P_discharge = action, 0.0
	        else:
	        	P_charge, P_discharge = 0.0, action
		
		P_grid = state[0] + P_charge + P_discharge

		return P_grid

	def getNextState(self, episode_number, time_step, state_k, action_k) :
		#print "state in getNextState",state_k
		current_netload = state_k[0]
		current_energy = state_k[1]
	
		if action_k >= 0:
			P_charge, P_discharge = action_k, 0.0
	    	else:
	        	P_charge, P_discharge = 0.0, action_k
			
	        E_next = current_energy + self.eta * P_charge + P_discharge
	        P_grid = current_netload + P_charge + P_discharge
		isValid = (P_grid < 0)
		reward  = - P_grid*self.getPrice(time_step)
		
		if isValid:
			reward = -10
			nextState = None

		price = self.getPrice(time_step+1)
		nextState = [self.getNetload(episode_number, time_step+1), E_next, price, time_step+1]
		#print "nextstate after ",nextState
		return nextState, reward, isValid
	
	
	def getMaxQvalue(self, state, agent) :
		legalActions = agent.getLegalActions(state, self)
		flag=0             
		QValue=None
		optimalAction=None
		for action in legalActions:
			if flag==0:
				QValue=model.predictQvalue(state,action)
				optimalAction=action
				flag=1
	            	else:
				if QValue < model.predictQvalue(state, action):
					QValue = model.predictQvalue(state, action)
					optimalAction=action
        	return [optimalAction], QValue
	
	def getNetload(self, episode_number, timeStep) :
		"change here by Siddharth "
		if timeStep > 23 :
			episode_number += 1
			episode_number %= self.day_chunk*self.training_time
			timeStep %= 24
		return self.net_load[episode_number][timeStep]

	
	def getPrice(self, timeStep) :
		#price = [.040,.040,.040,.040,.040,.040,.080,.080,.080,.080,.040,.040,.080,.080,.080,.040,.040,.120,.120,.040,.040,.040,.040,.040]
		#price = [.040,.040,.080,.080,.120,.240,.120,.040,.040,.040,.040,.080,.120,.080,.120,.040,.040,.120,.120,.040,.040,.040,.040,.040]
		#price = [.040,.040,.040,.040,.040,.040,.040,.040,.040,.040,.040,.040,.040,.040,.040,.040,.040,.080, .080,.120,.120,.040,.040,.040]
		if timeStep > 23 :
			timeStep %= 24
		return self.price_scheme[timeStep]

	def getSolar(self, episode_number,timeStep):
		if timeStep > 23 :
			episode_number += 1
			episode_number %= self.day_chunk*self.training_time
			timeStep %= 24
		solar_chunk = self.df_solar.ix[self.start:self.start+self.day_chunk-1].reset_index(drop=True)
		return solar_chunk.ix[episode_number][timeStep]

	def getLoad(self, episode_number, timeStep):
		if timeStep > 23 :
			episode_number += 1
			episode_number %= self.day_chunk*self.training_time
			timeStep %= 24
		load_chunk = self.df_load.ix[self.start:self.start+self.day_chunk-1].reset_index(drop=True)
		return load_chunk.ix[episode_number][timeStep]
	
	
