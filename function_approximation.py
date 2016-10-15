import numpy as np 
from sklearn import linear_model
from sklearn.svm import SVR
from sklearn import tree
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import ExtraTreesRegressor 
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils import shuffle
from sklearn.kernel_approximation import RBFSampler
import sklearn.pipeline


class FunctionApproximation:
    '''
    Training algorithm: Variety of algorithms (linear, ridge, tree based, kernelized) used for representing the Q(s,a). 

    Feature Extraction: Construct kernelized features (RBF, Fourier etc) from state variables for capturing non-linearity

    Batches: FittedQIteration alternates between constructing batches of experience and then fitting the Q-function.

    Note: parameters/weights correspond to the chosen action in the given state

    '''

    def __init__(self, name, actions):
        self.debug = True
        self.actions = actions
        self.initial_data = np.array([[3, 3.80, 0.40, 1.50], [3.0783744000000004, 1.7999999999999998, 0.04, 2.5], 
                                    [3.6603792000000004, 5.8500000000000005, 0.08, -1.5], [2.8383936000000003, 5.8500000000000005, 0.04, -3.0],
                                    [4.5679104000000015, 5.8500000000000005, 0.04, -2.0], [2.885976, 4.05, 0.04, 1.0]])

        self.initial_labels = np.array([[-1.0], [-0.2], [-0.5], [-.25], [0.0], [-2.1]])
        self.model_name = str(name)
        self.observation_samples = np.array([self.sample_states() for obv in range(10000)])
        self.featurizer = sklearn.pipeline.FeatureUnion([("rbf1", RBFSampler(gamma=5.0, n_components=100)),
                                                         ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
                                                         ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
                                                         ("rbf4", RBFSampler(gamma=0.5, n_components=100))])
        self.featurizer.fit(self.observation_samples)
        self.feature_scaler = StandardScaler()
        self.feature_scaler.fit(self.observation_samples)

        if self.model_name == 'svr':
                self.model = SVR(kernel='rbf')
                #self.model.fit(self.initial_data, self.initial_labels)

        elif self.model_name == 'extra_trees':    
                self.model = ExtraTreesRegressor().fit(self.initial_data, self.initial_labels)

        elif self.model_name == 'sgd':
            self.models = {}
            for a in range(len(self.actions)):
                model = SGDRegressor(learning_rate="constant")
                model.partial_fit([ self.featurize_state([3.6603792000000004, 2.8500000000000005, 0.08]) ], [0])
                self.models[a] = model
            #self.model = SGDRegressor(penalty='none')
            #self.model.fit(self.feature_scaler.transform(self.initial_data), self.initial_labels)
        else:
            self.model = None

    def sample_states(self):
        '''
        state = [netload, energy, price]
        '''
        load_sample = np.random.uniform(low=-2.6, high=6.0)
        energy_sample = np.random.uniform(low=1.7, high=6.0)
        price_sample = np.random.uniform(low=0.0, high=0.12)
        sample = [load_sample, energy_sample, price_sample]
        return sample

    def featurize_state(self, state):
        '''
        1. Normalize the observation/state-space with zero mean and unit variance.
        2. Return RBF kernelzied features #CHECK SUTTON BARTO SECTION for intuition
        '''
        scaled = self.feature_scaler.transform(state)
        featurized = self.featurizer.transform(scaled)
        return featurized[0]
    
    def update_qfunction(self, state, action_index, qvalue):
            features = self.featurize_state(state)
            self.models[action_index].partial_fit([features], [qvalue])
            return None
    '''
    def update_qfunction(self, minibatch, agent_instance):
        #Update the model/estimator/FA for each action.
        minibatch = np.array(minibatch) # (netload, energy, price, qvalue, action_index)
        groups = np.split(minibatch, np.where(np.diff(minibatch[:,4]))[0]+1) #returns a list with group of array elements
        actions = np.unique([x[0][4]for x in groups]) # unqiue actions
        print len(actions)
        for i, action_index in enumerate(actions):
            features = self.featurize_state(groups[i][:,:3])
            qvalue = groups[i][:,3:4]
            #print 'features', features.shape
            #print 'qvalues', qvalue
            #print 'obvs', self.observation_samples.shape
            self.models[action_index].partial_fit([features], qvalue)
    '''
    def predictQvalue(self, state, agent_instance, legal_actions):

            features = self.featurize_state(state)
            #print'predict', features.shape
            if len(legal_actions) != 1:
                pred = np.array([ self.models[key].predict([features])[0] for key in legal_actions])
                return np.max(pred)
            else:
                key = legal_actions[0]
                return self.models[key].predict([features])[0]
'''
    def predictQvalue(self, state, agent_instance, legal_actions):

            #stateaction = numpy.zeros((len(self.actions), len(state)))
            #stateaction[action,:] = state
        qvalues = []

        if len(legal_actions) == 1: 
            action_index = legal_actions[0]
            action = agent_instance.actions[action_index]
            features = state + [action]
            features = np.array(features)  
            features_scaled = self.feature_scaler.transform(features)
            qvalue = self.model.predict(features)
            #qvalue = self.model.predict(features_scaled)
            return qvalue[0]

        else:
            #print 'not inside 1:', len(legal_actions)
            for action_index in legal_actions:
                action = agent_instance.actions[action_index]
                features = state + [action]
                features = np.array(features)  
                features_scaled = self.feature_scaler.transform(features)
                prediction = self.model.predict(features)
                #prediction = self.model.predict(features_scaled)
                qvalues.append(prediction[0].tolist())
                
                if len(qvalues) == 0:
                    print 'Zero length, state', state
            
            return np.max(qvalues)

    def update_qfunction(self, minibatch, agent_instance):
        #print 'Weights before update',self.model.coef_
     
        X_train = [] #features (states? actions? both?)
        y_train = [] #qvalues

        #Loop through the batch and create the training set.

        for memory in minibatch:

            currentStateAction, kQvalue = memory

            #Get prediction of Q(currentState, currentAaction)
            #Ideally input features???
            
            X_train.append(np.asarray(currentStateAction))
            y_train.append(np.asarray(kQvalue))

        X_train = np.array(X_train)
        y_train = np.array(y_train).reshape(400,1)
        #print X_train.shape, y_train.shape

        #Shuffling for stochastic gradient descent and breaking correlations
        X_train, y_train = shuffle(X_train, y_train)
        #print X_train, y_train

        #Standardize the features
        X_scaled = self.feature_scaler.fit_transform(X_train)
        
        #Now update the model or Q-function
        #self.model.partial_fit(X_scaled, y_train)
        self.model.fit(X_train, y_train)
        #self.model.fit(X_scaled, y_train)
'''