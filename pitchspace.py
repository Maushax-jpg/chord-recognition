import numpy as np
import utils
from librosa.feature import tonnetz
import features

def sym(n,g):
    '''SYM-operator described by Gatzsche and Mahnert'''
    try:   
        return int(np.mod((n+(g/2)),g)-(g/2))
    except ValueError:
        return None 
    
def sym3(n,g,n0):
    '''three parameter SYM-operator described by Gatzsche and Mahnert'''
    x = sym(n-n0,g)
    if x is not None:
        return x+n0
    else:
        return None

def computeCPSSfeatures(chroma):
    """compute circular pitch space system features
        returns a 37xN array""" 
    if chroma.ndim == 1:
        chroma = np.reshape(chroma, (1, 12))
    elif chroma.shape[0] != 12:
        raise ValueError("Array shape must be (12, X).")

    rho_F  = np.zeros((chroma.shape[1],),dtype=complex)
    rho_FR = np.zeros_like(chroma,dtype=complex)
    rho_TR = np.zeros_like(chroma,dtype=complex)
    rho_DR = np.zeros_like(chroma,dtype=complex)

    for key in utils.pitch_classes:  # iterate over pitch classes for key related circles
        n_k = key.num_accidentals
        key_index = key.pitch_class_index
        for pitch in utils.pitch_classes: # iterate over all chroma bins
            n_f = sym3(49*pitch.chromatic_index,84,7*n_k) # index in sequence of perfect fifths
            if n_k == 0:
                rho_F += chroma[pitch.pitch_class_index,:] * np.exp(-1j*2*np.pi*(n_f/84)) * 1j
            if -21 <= (n_f-7*n_k) < 21:
                n_fr = sym(n_f-7*n_k, 48) # index in key related circle of fifths
                rho_FR[key_index,:] += chroma[pitch.pitch_class_index,:] * np.exp(-1j*2*np.pi*(n_fr/48)) * 1j
                
                # third related circle  
                n_tr = sym(n_f-7*n_k-12,24)  # index in key related circle of thirds
                rho_TR[key_index,:] += chroma[pitch.pitch_class_index,:] * np.exp(-1j*2*np.pi*(n_tr/24)) * 1j     
                # diatonic circle   
                n_dr = sym(n_f-7*n_k,12)  # index in key related diatonic circle
                rho_DR[key_index,:] += chroma[pitch.pitch_class_index,:] * np.exp(-1j*2*np.pi*(n_dr/12)) * 1j

    F = np.zeros((2,chroma.shape[1]),dtype=float)
    FR = np.zeros((24,chroma.shape[1]),dtype=float)
    TR = np.zeros((24,chroma.shape[1]),dtype=float)
    DR = np.zeros((24,chroma.shape[1]),dtype=float)
    F[0,:] = rho_F.real
    F[1,:] = rho_F.imag
    for i,key_index in zip(np.arange(0,24,2),range(12)):
        FR[i  ,:] = rho_FR[key_index,:].real
        FR[i+1,:] = rho_FR[key_index,:].imag
    for i,key_index in zip(np.arange(0,24,2),range(12)):
        TR[i  ,:] = rho_TR[key_index,:].real
        TR[i+1,:] = rho_TR[key_index,:].imag
    for i,key_index in zip(np.arange(0,24,2),range(12)):
        DR[i  ,:] = rho_DR[key_index,:].real
        DR[i+1,:] = rho_DR[key_index,:].imag
    return F,FR,TR,DR

def computeHCDF(chroma,prefilter_length=7,use_cpss=True):
    """
    harmonic change detection function
    """
    if use_cpss:
        F,FR,TR,DR = computeCPSSfeatures(chroma)
        pitchspace_features = np.vstack((F, FR, TR, DR ))
    else:
        pitchspace_features = tonnetz(chroma=chroma)

    # apply prefilter and zero pad 
    pitchspace_features = features.applyPrefilter(None,pitchspace_features,filter_type="median",prefilter_length=prefilter_length)
    pitchspace_features = np.pad(pitchspace_features, ((0, 0), (1, 1)), mode='constant', constant_values=(0, 0))

    # compute harmonic change detection functions
    harmonic_change = np.sqrt(
        np.sum(
            np.square(pitchspace_features[:, :-2] - pitchspace_features[:, 2:])
            ,axis=0
        )
    )
    return harmonic_change
