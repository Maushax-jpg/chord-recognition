from hmmlearn import hmm
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dataloader 


# train a HMM and save it to pickle file

chord_labels_majmin = {'N':0,'C':1,'C#':2,'D':3,'D#':4,'E':5,'F':6,'F#':7,'G':8,'G#':9,'A':10,'A#':11,'B':12,
    'C:min':13,'C#:min':14,'D:min':15,'D#:min':16,'E:min':17,'F:min':18,'F#:min':19,'G:min':20,'G#:min':21,
    'A:min':22,'A#:min':23,'B:min':24
}
chord_index_majmin = {value: key for key, value in chord_labels_majmin.items()}


def find_changes(lst):
    """returns list of indices where the list elements differ"""
    changes = [0]
    for i in range(1, len(lst)):
        if lst[i] != lst[i-1]:
            changes.append(i)
    return changes

def postprocessing(predictions):
    index = [i for i in find_changes(predictions)]
    chord_sequence = [chord_index_majmin[predictions[x]] for x in index]
    return (index,chord_sequence)

def getChordStatistics(chroma_df,dataset='beatles'):
    halftones = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    mu = []
    cov_matrices = []
    for i,label in enumerate(chord_labels_majmin):
        temp = chroma_df.loc[chroma_df['label'] == label]
        mu.append(temp[halftones].mean().values)
        cov_matrices.append(temp[halftones].cov())
    return mu,cov_matrices

def getInitialStateProbability(dataset='beatles'):
    # equal distribution
    initial_state_probability = np.ones((25,),dtype=float)/len(chord_labels_majmin)
    return initial_state_probability

def getTransitionProbability(labels_df,dataset='beatles',drop_consecutives=True,plot=True):
    if dataset == 'beatles':
        if drop_consecutives:
            # drop consecutive duplicates by comparison  
            labels_df = labels_df.loc[labels_df.shift(-1) != labels_df]
        transitions = np.zeros((25,25),dtype=float)+np.finfo(float).eps
        for x,y in zip(labels_df,labels_df.shift(-1)):
            try:
                transitions[chord_labels_majmin[x],chord_labels_majmin[y]] += 1.0
            except KeyError:
                continue
        for row in range(transitions.shape[0]):
            transitions[row,:] = transitions[row,:]/np.sum(transitions[row,:])
        transition_probability = transitions
        if plot:
            fig,ax = plt.subplots()
            ax.imshow(transition_probability)
            ax.set_xticks(np.linspace(0,24,25))
            ax.set_yticks(np.linspace(0,24,25))
            ax.set_xticklabels(list(chord_labels_majmin.keys()),rotation=-90)
            ax.set_yticklabels(list(chord_labels_majmin.keys()))
        return transition_probability


if __name__ == '__main__': 
    chroma_df = pd.read_pickle("/home/max/ET-TI/Masterarbeit/prototyping/data/beatles/beatles_chroma.pkl")
    chroma_majmin_df = pd.read_pickle("/home/max/ET-TI/Masterarbeit/prototyping/data/beatles/beatles_chroma_majmin.pkl")  

    # calculate statistics for HMM 
    transitions = getTransitionProbability(chroma_majmin_df['label'],drop_consecutives=False)
    initialstates = getInitialStateProbability()
    mu,cov = getChordStatistics(chroma_majmin_df)

    # continuous emission model
    model = hmm.GaussianHMM(n_components=transitions.shape[0], covariance_type="full")

    # initial state probability
    model.startprob_ = initialstates
    # transition matrix probability
    model.transmat_ = transitions

    # part of continuous emission probability - multidimensional gaussian
    # 12 dimensional mean vector
    model.means_ = mu
    # array of covariance of shape [n_states, n_features, n_features]
    model.covars_ = cov

    # Save HMM model to a file
    joblib.dump(model, 'hmm_model.pkl')

