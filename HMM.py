import hmmlearn.hmm
import numpy as np
import mir_eval

class HiddenMarkovModel():
    """
    A Wrapper class for hmmlearn gaussianHMM 
        
        Parameters:
            @ train_data: Pandas dataframe containing labeled features
            @ states: a list of names for the model states

        Examples: 
        >>> train_data = pandas.DataFrame(columns=['C','C#',..,'B','label'])
        >>> states = ['C','Cm','C7']
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
                print(f"KeyError: {x} -> label ist not a valid state! skipping transition..")
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
        t_start =round(t[0],2)
        chord_changes = [self.state_index[predictions[0]]]
        intervals=[]
        for i in range(1, len(predictions)):
            if predictions[i] != predictions[i-1]:
                chord_changes.append(self.state_index[predictions[i]])
                intervals.append((t_start,round(t[i],2)))
                t_start = round(t[i],2)
        return intervals,chord_changes
        
