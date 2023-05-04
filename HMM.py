import hmmlearn.hmm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import joblib
import os
import circularPitchSpace as cps

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
        self.mean_vectors = self.calculateChromaMean(train_data)
        self.cov_matrices = self.calculateChromaCovariance(train_data)
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

    def calculateChromaMean(self,train_data):
        """
        calculate mean vector for every state
        """
        mean_vectors = []
        for state in self.state_labels:
            temp = train_data.loc[train_data['label'] == state]
            temp.drop(columns='label')
            mean_vectors.append(temp.mean(numeric_only=True).values)
        return mean_vectors

    def calculateChromaCovariance(self,train_data):
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
        remove = []
        for i in range(len(chord_changes)):
            if (t_stop[i]-t_start[i]) < 0.3:
                remove.append(i)
                if i > 1:
                    t_stop[i-1] = t_stop[i]
        chord_changes = [chord_changes[i] for i in range(len(chord_changes)) if i not in remove]
        t_start = [t_start[i] for i in range(len(t_start)) if i not in remove]
        t_stop = [t_stop[i] for i in range(len(t_stop)) if i not in remove]
        # shape = (n_events, 2)
        intervals = np.array([t_start,t_stop],dtype=float).T
        return intervals,chord_changes
        
    def save(self,path,filename):
        # Save HMM model to a file
        joblib.dump(self, os.path.join(path,filename))

    def plotStatistics(self,state='C:major'):
        try: 
            index = self.state_labels[state]
        except KeyError:
            print(f"Model has no state: {state}")
            return
        ticks = [0,2,4,5,7,9,11]
        labels = ['C','D','E','F','G','A','B']
        fig,ax = plt.subplots()
        img = ax.imshow(self.cov_matrices[index])
        fig.colorbar(img)
        ax.set_title(f"Cov {state}")
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels)
        plt.show()
        fig,ax = plt.subplots()
        ax.set_title('mean Chroma energy D:maj')
        ax.bar(np.linspace(0,11,12),self.mean_vectors[index])
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)
        plt.show()
        fig,ax = plt.subplots()
        img = ax.imshow(self.transition_probability)
        print(np.diag(self.transition_probability))
        fig.colorbar(img)   
        plt.show()

def load_model(path):
    """load model from .pkl file
        NOT IMPLEMENTED: ERROR checking etc.
    """
    model = joblib.load(path)
    return model

if __name__ == '__main__':
    model = load_model('/home/max/ET-TI/Masterarbeit/models/hmm_model.pkl')
    chroma = np.zeros((4,12),dtype = float)

    # define colors and chords for visualization in Pitch space 
    chords = ["C:maj",'C:min','A:maj','A:min']
    colors = ["r","y","b","k"]

    for i,(chord,color) in enumerate(zip(chords,colors)):
        
        chroma[i,:] = np.array(model.mean_vectors[model.state_labels[chord]],dtype=float)
    r_F,r_FR,r_TR,r_DR = cps.transformChroma(chroma)
    fig,ax_list = cps.plotPitchSpace()
    for i in range(len(chords)):
        cps.plotFeatures(ax_list,r_F[i],r_FR[i],r_TR[i],r_DR[i],color=colors[i])
    #create legend
    legend=[]
    for color,name in zip(colors,chords):
        legend.append(matplotlib.lines.Line2D([0], [0], color=color, lw=2, label=f"{name}"))
    # plot legend
    ax_list[0][5].legend(handles=legend, loc='center')
    ax_list[0][0].text(0.5,0.5,"Circular Pitch Space",fontsize=13)
    plt.show()
    
