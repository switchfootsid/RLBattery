from copy import deepcopy

class Environment :
	def __init__(self, currentState, Gamma, Eta, day_chunk, total_years) :
		"change here by Siddharth "
		self.Eta = Eta #(battery efficiency)
		self.Gamma = Gamma
		self.currentState = currentState
		self.training_time = day_chunk*total_years #change here by Siddharth
		self.diff = (df_load.ix[0:self.day_chunk-1] - df_solar.ix[0:self.day_chunk-1]) #change here by Siddharth, just take first 15 days 
    	self.net_load = pd.concat([self.diff]*self.training_time, ignore_index=True).values.tolist()
	
	def next_step(self, action_sequence, k, FA, agent) :
		'''
        Perform constraint checking (energy, grid) and assign penalty/rewards.
        Output: reward/penalty, next state, constraint satisfaction (boolean)
        '''
		cumulativeReward, lastState = self.getCumulativeReward(k, action_sequence)
        self.currentState, reward, isValid = self.getNextState(self.currentState, action_sequence[0])
        qValueLastState = FA.predictQvalue(lastState, agent.getLegalActions(lastState, self))
		return currentState, cumulativeReward + (self.Gamma**3)*qValueLastState, isValid
	
	def getCumulativeReward(self, k, actions) :
		gamma = 1
		state = deepcopy(currentState)
		cr = 0
		for i in range(0, k) :
			state, reward, isValid = getNextState(state, actions[i])
			if(not isValid) :
				return None
			cr += gamma*reward 
			gamma *= self.Gamma
		return state, cr # target = (R1 + gamma*R2 + gamma2*Qmax
	
	def getNextState(self, state_k, action_k) :
	
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
        return [optimalAction],QValue
	
	def getNetload(self, timeStep) :
		"change here by Siddharth "
		df_solar = pd.read_csv('./solar_clean.csv')
    	df_load = pd.read_csv('./load_data_peak6.csv')
    	#diff = (df_load.ix[0:self.day_chunk-1] - df_solar.ix[0:self.day_chunk-1]) #change here by Siddharth, just take first 15 days 
    	#net_load = pd.concat([diff]*self.training_time, ignore_index=True).values.tolist() #check format
    	
    	return net_load[timeStep-1]

	
	def getPrice(self, timeStep) :
		price = np.array([.040,.040,.040,.040,.040,.040,.080,.080,.080,.080,.040,.040,.080,.080,.080,.040,.040,.120,.120,.040,.040,.040,.040,.040]).tolist()
		return price[timeStep-1]
		
	
	