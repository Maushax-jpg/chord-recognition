import hmmlearn.hmm
import numpy as np
import joblib
import os

class HiddenMarkovModel():
    """
    A Wrapper class for hmmlearn gaussianHMM 
        
        Parameters:
            @ train_data: Pandas dataframe containing labeled features
            @ states: a list of names for the model states

        Examples: 
        >>> train_data = pandas.DataFrame(columns=['C','C#',..,'B','label'])
        >>> states = ['C','C:min',..,'B','B:min']
        >>> HMM = HiddenMarkovModel(train_data,states)
    """
    def __init__(self,train_data,state_labels,pi_matrix=None):
        # mapping of chord label to index and vice versa with two dictionaries
        self.state_labels = dict(zip(state_labels,range(len(state_labels))))
        self.state_index = dict(zip(range(len(state_labels)),state_labels))
        self.mean_vectors = self.calculateMean(train_data)
        self.cov_matrices = self.calculateCovariance(train_data)
        self.initial_state_probability = self.getInitialStateProbability(pi_matrix)
        self.transition_probability = self.getTransitionProbability(train_data)
        self.model = self.buildModel()

    def getInitialStateProbability(self,pi_matrix=None):
        if pi_matrix is None:
            # assume equal distribution
            return np.ones((len(self.state_labels),),dtype=float)/len(self.state_labels)
        else:
            return pi_matrix
    # equal distribution

    def getTransitionProbability(self,train_data):
        transitions = np.zeros((len(self.state_labels),len(self.state_labels)),dtype=float)+np.finfo(float).eps
        for x,y in zip(train_data['label'],train_data['label'].shift(-1)):
            try:
                transitions[self.state_labels[x],self.state_labels[y]] += 1.0
            except KeyError:
                print(f"KeyError: Transition:{x,y} -> label ist not a valid state! skipping transition..")
                continue
        # normalize transitions to get probability
        for row in range(len(self.state_labels)):
            transitions[row,:] = transitions[row,:]/np.sum(transitions[row,:])
        return transitions

    def calculateMean(self,train_data):
        """
        calculate mean vector for every state
        """
        mean_vectors = []
        for state in self.state_labels:
            temp = train_data.loc[train_data['label'] == state]
            temp.drop(columns='label')
            mean_vectors.append(temp.mean(numeric_only=True).values)
        return mean_vectors

    def calculateCovariance(self,train_data):
        """
        get covariance matrices for all features from labeled feature vector
            shape: [n_states, n_features, n_features]
        """
        cov_matrices = []
        for state in self.state_labels:
            temp = train_data.loc[train_data['label'] == state]
            cov_matrices.append(temp.cov(numeric_only=True))
        return cov_matrices
    
    def evaluate(self,test_data):   
        raise NotImplementedError

    def buildModel(self):
        # continuous emission model
        model = hmmlearn.hmm.GaussianHMM(n_components=len(self.state_labels), covariance_type="full")

        # initial state probability
        model.startprob_ = self.initial_state_probability
        # transition matrix probability
        model.transmat_ = self.transition_probability
        # 12 dimensional mean vector
        model.means_ = self.mean_vectors
        model.covars_ = self.cov_matrices
        return model
    
    def predict(self,t,features):
        predictions = self.model.predict(features)

        # initialize chord changes with first prediction
        chord_changes = [self.state_index[predictions[0]]]
        t_start = [t[0]]
        t_stop = []
        for i in range(1, len(predictions)):
            if predictions[i] != predictions[i-1]:
                t_stop.append(round(t[i],2))
                # add chord change
                chord_changes.append(self.state_index[predictions[i]])
                t_start.append(round(t[i],2))
        # append last t_stop to conclude predictions
        t_stop.append(round(t[-1],2))
        return t_start,t_stop,chord_changes
        
def save_model(path,HiddenMarkovModel):
    # Save HMM model to a file
    joblib.dump(HiddenMarkovModel, os.path.join(path,'hmm_model.pkl'))

def load_model(path):
    """load model from .pkl file
        NOT IMPLEMENTED: ERROR checking etc.
    """
    model = joblib.load(path)
    return model