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

    def computeLogLikelihood(self,x):
        """evaluates the class log-likelihood function at the observation x"""
        # pdf of a multivariate normal distribution evaluated at the observation x 
        #  (2pi)^(-k/2)*det(Cov)^(-1/2)*exp(-1/2*(x-mu)^T * Cov^(-1) * (x-mu))
        temp = ((x - self._mu).T @ np.linalg.inv(self._cov)) @ (x - self._mu)
        logpdf = -(self._x.shape[0]/2) * np.log(2*np.pi) - 0.5 * np.log(np.linalg.det(self._cov)) -0.5*temp
        if logpdf < 0:
            pass
        return logpdf

class CPSS_Classifier():
    def __init__(self,alphabet="majmin",filepath="/home/max/ET-TI/Masterarbeit/chord-recognition/models/cpss",split=1): 
        self._labels = utils.createChordTemplates(alphabet)[1]
        self._key_templates = utils.createKeyTemplates()
        self._model = self.loadChordmodel(filepath, alphabet, split)

    def loadChordmodel(self,filepath,alphabet,split):
        path = f"{filepath}/cpss_{alphabet}_dcp.npy"
        return np.load(path,allow_pickle=True)[()]

    def computeLikelihoods(self,chromavector,key_index):
        likelyhood = np.zeros((len(self._labels),), dtype=float)
        # compute feature vector for the selected key
        F,FR,TR,_ = computeCPSSfeatures(chromavector)
        x = (
            F[0,:], F[1,:],
            FR[2 * key_index, :], FR[2*key_index+1, :],
            TR[2 * key_index, :], TR[2*key_index+1, :]
        )
        # compute likelyhood
        for chordmodel in self._model[key_index]:
            likelyhood[chordmodel._index] = chordmodel.computeLogLikelihood(x)
        return likelyhood
    
    def selectKey(self, chromavector):
        inner_product = np.matmul(self._key_templates.T, chromavector)
        return np.argmax(inner_product)

    def classify(self, chromavector,key_index):
        likelyhood = self.computeLikelihoods(chromavector,key_index=key_index)
        index = np.argmax(likelyhood)
        # negative likelyhood -> this occurs if there is no good fit for the chromavector
        if likelyhood[index] > 0:
            return index
        else:
            return None