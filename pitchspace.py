import numpy as np
import utils
from librosa.feature import tonnetz
import features
import train
from scipy.signal import find_peaks

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
        chroma = np.reshape(chroma, (12, 1))
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

def computeHCDF(chroma,prefilter_length=3,use_cpss=True):
    """
    harmonic change detection function
    """
    if use_cpss:
        F,FR,TR,DR = computeCPSSfeatures(chroma)
        pitchspace_features = np.vstack((F,FR,TR))
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

def computeKeyCorrelation(chroma):
    """ calculate correlation with major key profiles (krumhansl).
    returns the most likey key for every time instance"""
    templates = np.zeros((12,12),dtype=float)
    key_profile = np.array([5, 2, 3.5, 2, 4.5, 4, 2, 4.5, 2, 3.5, 1.5, 4])
    ic_energy = np.zeros_like(chroma)    
    for i in range(12):
        templates[i, :] = np.roll(key_profile,i)
        # neglect all chromabins that are not part of the diatonic circle -> key_related_chroma
        pitch_classes = np.roll([1,0,1,0,1,1,0,1,0,1,0,1],i)
        key_related_chroma = np.multiply(chroma, np.reshape(pitch_classes, (12,1))) 
        ic_energy[i, :] = np.sum(computeIntervalCategories(key_related_chroma), axis=0)
 
    correlation = np.matmul(templates,chroma) # matrix product (12x12) x (12 x N)
    correlation_energy = np.square(correlation) + ic_energy
    return correlation_energy

def computeIntervalCategories(chroma):
    if chroma.shape[0] != 12:
        raise ValueError("invalid Chromagram shape!")
    ic = np.zeros((6, chroma.shape[1]), dtype=float)

    for i in range(6):
        rotated_chroma = np.roll(chroma, shift=i+1, axis=0)
        ic[i,:] = np.sum(chroma * rotated_chroma, axis=0)
    return ic

class ChordModel:
    """class for storing multivariate statistic of a chord in the pitch space system.

       Features must be of size (x,N) where N are the number of observations
    """
    def __init__(self,index, features):
        self._index = index
        self._x = np.vstack(features)
        self._mu = np.mean(self._x,axis=1).reshape(self._x.shape[0],1)
        self._cov= self.estimateCovariance()
    
    def estimateCovariance(self):
        cov = np.zeros((self._x.shape[0],self._x.shape[0]),dtype=float)
        N = self._x.shape[1]
        for j in range(self._x.shape[0]):
            for k in range(self._x.shape[0]):
                temp = 0
                for i in range(N):
                    temp += (self._x[j, i] - self._mu[j]) * (self._x[k, i] - self._mu[k])
                cov[j,k] = temp / (N - 1)
        return cov

    def computeLogLikelyhood(self,x):
        """evaluates the class likelyhood function at the specific point for both circles"""
        # matrix product (x-mu)^T * Cov^(-1) * (x-mu)
        res = ((x - self._mu).T @ np.linalg.inv(self._cov)) @ (x - self._mu)
        return -0.5 * (np.log(np.linalg.det(self._cov)) + res + self._x.shape[0] * np.log(2*np.pi))

class Classifier():
    def __init__(self,filepath="/home/max/ET-TI/Masterarbeit/chord-recognition/models/chromadata_root_invariant_median.hdf5"): 
        self._labels = utils.createChordTemplates("sevenths")[1]
        self._model = self.createChordmodel(filepath)

    def createChordmodel(self,filepath):
        from tqdm import tqdm
        chromadata = train.loadChromadata(filepath)
        # offset in template array (see utils.createChordTemplates)
        offsets = {"maj":0,"min":12,"maj7":24,"7":36,"min7":48} 
        shift = [0,2,4,5,7,9]
        keys = {}
        for key_index in tqdm(range(12),desc=f"creating model for key"):
            chordmodels = []
            # create the triads for all major keys  
            for i,qual in zip(shift,["maj","min","min","maj","maj","min"]):
                chroma = np.roll(chromadata[qual],key_index + i,axis=0)
                # transform onto pitchspace 
                F,FR,TR,_ = computeCPSSfeatures(chroma)
                # rearange features in a tuple
                x = (
                    F[0,:], F[1,:],
                    FR[2 * key_index, :], FR[2*key_index+1, :],
                    TR[2 * key_index, :], TR[2*key_index+1, :]
                )
                chord_index = (key_index + i) % 12 + offsets[qual]
                chordmodels.append(ChordModel(chord_index, x))

            # create the tetrads for all major keys  
            for i,qual in zip(shift,["maj7","min7","min7","maj7","7","min7"]):
                chroma = np.roll(chromadata[qual],key_index + i,axis=0)
                # transform onto pitchspace 
                F,FR,TR,_ = computeCPSSfeatures(chroma)
                # rearange features in a tuple
                x = (
                    F[0,:], F[1,:],
                    FR[2 * key_index, :], FR[2*key_index+1, :],
                    TR[2 * key_index, :], TR[2*key_index+1, :]
                )
                chord_index = (key_index + i) % 12 + offsets[qual]
                chordmodels.append(ChordModel(chord_index, x))
            keys[key_index] = chordmodels
        return keys

    def computeLogLikelyhood(self,chromavector):
        likelyhood = np.zeros((len(self._labels),), dtype=float)
        F,FR,TR,_ = computeCPSSfeatures(chromavector)
        for key_index,key in self._model.items():
            x = (
                F[0,:], F[1,:],
                FR[2 * key_index, :], FR[2*key_index+1, :],
                TR[2 * key_index, :], TR[2*key_index+1, :]
            )
            for chordmodel in key:
                temp = chordmodel.computeLogLikelyhood(x)
                # only update likelyhood for a chord if it exceeds a previous calculation 
                if temp > likelyhood[chordmodel._index]:
                    likelyhood[chordmodel._index] = temp
        return likelyhood
    
    def classify(self,chromavector):
        likelyhood = self.computeLogLikelyhood(chromavector)
        index_ml = np.argmax(likelyhood)
        return self._labels[index_ml],likelyhood[index_ml]