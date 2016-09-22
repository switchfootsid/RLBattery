from copy import deepcopy
import pandas as pd

class Environment :
	def __init__(self, Gamma, Eta, day_chunk, total_years) :
		"change here by Siddharth "
		self.Eta = Eta #(battery efficiency)
		self.Gamma = Gamma
		self.df_solar = pd.read_csv('./solardata.csv')
	   	self.df_load = pd.read_csv('./load_data_peak6.csv')
		self.training_time = day_chunk*total_years #change here by Siddharth
		self.diff = (df_load.ix[0:self.day_chunk-1] - df_solar.ix[0:self.day_chunk-1]) #change here by Siddharth, just take first 15 days 
    		self.net_load = pd.concat([self.diff]*self.training_time, ignore_index=True).values.tolist()
    		self.currentState = None
		
	def setInitialState(self, epsiode_number, E_init):
		'''
		Set's the initialState (0th hour) for episode_number.
		episode_number 
		E_init - passed by agent class
		'''
		net_load = float(net_load.iloc[episode_number][0])
		energy_level = E_init
		price = self.getPrice(0)
		
		self.currentState = [net_load, energy_level, price]
		
	def next_step(self, time_step, action_sequence, k, FA, agent) :
		'''
	        Perform constraint checking (energy, grid) and assign penalty/rewards.
	        Output: reward/penalty, next state, constraint satisfaction (boolean)
	        '''
		cumulativeReward, lastState = self.getCumulativeReward(time_step, k, action_sequence)
	        self.currentState, reward, isValid = self.getNextState(time_step, self.currentState, action_sequence[0])
	        qValueLastState = FA.predictQvalue(lastState, agent.getLegalActions(lastState, self))
		return currentState, cumulativeReward + (self.Gamma**3)*qValueLastState, isValid
	
	def getCumulativeReward(self, time_step, k, actions) :
		gamma = 1
		state = deepcopy(currentState)
		cr = 0
		for i in range(0, k) :
			state, reward, isValid = getNextState(time_step, state, actions[i])
			time_step += 1
			if(not isValid) :
				return None
			cr += gamma*reward 
			gamma *= self.Gamma
		return state, cr # target = (R1 + gamma*R2 + gamma2*Qmax
	
	def getNextState(self, time_step, state_k, action_k) :
	
		current_netload = state_k[0]
        	current_energy = state_k[1]
		
		if action_k >= 0:
            		P_charge, P_discharge = action_k, 0.0
	        else:
	        	P_charge, P_discharge = 0.0, action_k
			
	        E_next = current_energy + self.eta * P_charge + P_discharge
	        P_grid = current_netload + P_charge + P_discharge
		isValid = (P_grid < 0)
		reward  = -P_grid*self.get_price(time_step)
			
		nextState = (P_grid, E_next)
		return (nextState, reward, isValid)
	
	
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
	
	def getNetload(self, timeStep) :
		"change here by Siddharth "
		return net_load[timeStep-1]

	
	def getPrice(self, timeStep) :
		price = [.040,.040,.040,.040,.040,.040,.080,.080,.080,.080,.040,.040,.080,.080,.080,.040,.040,.120,.120,.040,.040,.040,.040,.040]
		return price[timeStep-1]
		
	
	