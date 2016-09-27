import numpy as np 
from sklearn import linear_model
from sklearn.svm import SVR
from sklearn import tree

class FunctionApproximation:
    '''
    Training algorithm: Variety of algorithms (linear, ridge, tree based, kernelized) used for representing the Q(s,a). 

    Feature Extraction: Construct kernelized features (RBF, Fourier etc) from state variables for capturing non-linearity

    Batches: FittedQIteration alternates between constructing batches of experience and then fitting the Q-function.

    Note: parameters/weights correspond to the chosen action in the given state

    '''

    def __init__(self, name):
        self.debug = True
        self.model_name = str(name)
        #self.model = model
        #self.kernel = str(kernel)

        if self.model_name == 'linear':
            self.model = linear_model.LinearRegression()

        elif self.model_name == 'ridge':    
            self.model = linear_model.Ridge(alpha='l2')

        elif self.model_name == 'svr':
            self.model = SVR(kernel='rbf', C=1e3, gamma=0.1)
            self.model.fit(np.array([2.3, 1.3, 0.040, 1.00]).reshape(1,4), np.array([-1.5]).reshape(1,-1))
        else:
            self.model = None

    def feature_extraction(self, state, action):
        '''
        * NOT USING currently *
        - Return a basis or prjection of variable into high dimensional space, kernelize features. 
        - This is similar to Tile Coding or RBF networks. 

        param state: state variables
        param action: chosen action

        returns: features/basis vectors for state variables

        '''

        if kernel == 'rbf':
            features = self.RBF_features(state)
            stateaction = np.zeros((self.actions, len(state)))
            stateaction[action,:] = state
            return stateaction.flatten()

        elif kernel == 'tile_coding':
            features = self.CMAC_features(state)
        else:
            features = state

        return features

    def CMAC_features(self, state, action):
        '''
        * NOT USING CURRENTLY * 
        - Copy code from Rich Sutton's website.
        state: state variables
        return: number of on tiles for the state space.
        '''
        pass

    def predictQvalue(self, state, agent_instance, legal_actions):
        '''
        - Get the best Q-value function for the greedy action in State 

        param state: state variables (currentState, nextState)
        param legal_actions: set of actions allowed in the state

        return: maxQ(state, a) for a in legal_actions.
        '''

            #stateaction = numpy.zeros((len(self.actions), len(state)))
            #stateaction[action,:] = state
        qvalues = []

        if self.debug and len(legal_actions)==0:
                    file=open('debug.txt','a')
                    file.write('legalActions should not be empty in getAction in LearningAgent.py.......\n')
                    file.write(" state = " + str(state) + "\n") 
                    file.close()
        #print 'in predictQvalue ' + str(state)
        #print("legal_actions in fa are",legal_actions)

        for action_index in legal_actions:
            action = agent_instance.actions[action_index]
            features=state+[action]
            features = np.array(features).reshape(1,4)
            #print state
            prediction = self.model.predict(features)
            qvalues.append(prediction[0].tolist())

        if len(qvalues) == 0:
        	print 'Zero length, state', state
        return np.max(qvalues)

    def update_qfunction(self, minibatch, agent_instance):
        '''
        Fits the Q_function or the regression model to the experience or batch.

        - Minibatch: (state_action_features, Q_targets) Vs (S, A, R, S1)
        
        param minibatch: random sample of a experience/batch. 
        param agent_instance = agent object for accessing methods. 
        return: None
        '''
        X_train = [] #features (states? actions? both?)
        y_train = [] #qvalues

        #Loop through the batch and create the training set.

        for memory in minibatch:
            #get the stored values first.

            #currentState, currentAction, reward, nextState, kState = memory # currentState, kQvalue
            #print("memory is ",memory)
            currentStateAction, kQvalue = memory

            #Get prediction of Q(currentState, currentAaction)
            #Ideally input features???
            '''
            if kQvalue <= -100:
            #conditonal checking for invalid state
                target = kQvalue
            else: 
        
                #legal_next_actions = agent_instance.getLegalActions(nextState)
            
                next_qvals = [] #defaultdict or np.zeros ?????
                
                for action_index in legal_next_actions:
                    ##How to represent actions? Index or Numerical? 
                    action = agent_instance.actions[action_index]
                    features = np.asarray(currentState.append(action))
                    new_qval = self.model.predict(features)
                    next_qvals.append(new_qval)
                    https://github.com/switchfootsid/RLBattery.git
                best_qval = next_qvals.max()
                
                #target = kQvalue + agent_instance.gamma * kQvalue

            #currentSA = currentState.append(action)
            '''
            X_train.append(np.asarray(currentStateAction))
            y_train.append(np.asarray(kQvalue))

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        #Now update the model or Q-function
        self.model.fit(X_train, y_train)