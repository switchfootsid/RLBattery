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
        #self.initial_data = np.array([self.sample_states() for obv in range(100)])
        #self.initial_labels = np.array([0 for obv in range(100))
        self.model_name = str(name)
        self.observation_samples = np.array([self.sample_states() for obv in range(10000)])
        self.feature_scaler = StandardScaler()
        self.feature_scaler.fit(self.observation_samples)
        self.featurizer = sklearn.pipeline.FeatureUnion([("rbf1", RBFSampler(gamma=5.0, n_components=100)),
                                                         ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
                                                         ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
                                                         ("rbf4", RBFSampler(gamma=0.5, n_components=100))])
        self.featurizer.fit(self.feature_scaler.transform(self.observation_samples))

        if self.model_name == 'svr':
                self.model = SVR(kernel='rbf')
                #self.model.fit(self.initial_data, self.initial_labels)

        elif self.model_name == 'extra_trees':    
                self.models = {}
                for a in range(len(self.actions)):
                    model = ExtraTreesRegressor(random_state=42, oob_score=True, bootstrap=True) ###check hyperparameters
                    model.fit(self.featurize_state(self.observation_samples), np.full(len(self.observation_samples), -1))
                    #model.fit(self.observation_samples, np.full(len(self.observation_samples), -1))
                    self.models[a] = model

        elif self.model_name == 'sgd':
            self.models = {}
            for a in range(len(self.actions)):
                model = SGDRegressor(learning_rate="constant")
                model.partial_fit([self.featurize_state([3.6603792000000004, 2.8500000000000005, 0.08, 2])[0]], [0])
                self.models[a] = model


    def sample_states(self):
        '''
        state = [netload, energy, price]
        '''
        load_sample = np.random.uniform(low=-2.6, high=6.0)
        energy_sample = np.random.uniform(low=1.7, high=6.2)
        price_sample = np.random.uniform(low=0.0, high=0.12)
        time_sample = np.random.randint(low=0, high=25)
        #sample = [load_sample, energy_sample, price_sample]
        sample = [load_sample, energy_sample, price_sample, time_sample]
        return sample

    def featurize_state(self, state):
        '''
        1. Normalize the observation/state-space with zero mean and unit variance.
        2. Return RBF kernelzied features #CHECK SUTTON BARTO SECTION for intuition
        '''
        scaled = self.feature_scaler.transform(state)
        featurized = self.featurizer.transform(scaled)
        return featurized
    '''
    def update_qfunction(self, state, action_index, qvalue):
            features = self.featurize_state(state)[0]
            self.models[action_index].partial_fit([features], [qvalue])
            return None
    '''
    def update_qfunction(self, batch, targets):
        #Perform batch update for FQI

        for action_index in batch.iterkeys():
            minibatch = batch[action_index]
            features = self.featurize_state(minibatch)
            #features = minibatch
            qvalues = targets[action_index]
            self.models[action_index].fit(features, qvalues)
        
        #return None
    
    def predictQvalue(self, state, legal_actions):
            #print 'idhar bro',legal_actions, state

            features = self.featurize_state(state)[0]
            #features = state
            #print'predict', features.shape
            if len(legal_actions) != 1:
                pred = np.array([self.models[key].predict([features])[0] for key in legal_actions])
                return np.max(pred)
            else:
                key = legal_actions[0]
                return self.models[key].predict([features])[0]
