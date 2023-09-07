import numpy as np
import librosa
import madmom
from scipy.fftpack import dct,idct

def rms(y,hop_length=1024):
    """compute RMS value from mono audio signal"""
    return librosa.feature.rms(y=y,hop_length=hop_length//2).flatten()

def madmom_beats(audiopath,beats_per_bar=2):
    beat_processor = madmom.features.downbeats.DBNDownBeatTrackingProcessor(beats_per_bar,fps=100)
    activation_processor =  madmom.features.downbeats.RNNDownBeatProcessor()
    activations = activation_processor(audiopath)
    beats = beat_processor(activations)
    downbeats = [beat[0] for beat in beats if beat[1] == 1]
    upbeats = [beat[0] for beat in beats if beat[1] == 2]
    return downbeats,upbeats

def sumChromaDifferences(chroma):
    if chroma.shape[1] != 12:
        raise ValueError("invalid Chromagram shape!")
    # rearange chroma vector
    cq_fifth = np.zeros_like(chroma,dtype=float)
    for q in range(12):
        cq_fifth[:,q] = chroma[:,(q * 7) % 12]

    gamma_diff = np.zeros_like(chroma,dtype=float)
    for q in range(12):
        gamma_diff[:,q] = np.abs(cq_fifth[:,(q+1) % 12] - cq_fifth[:,q]) 
    # normalize to one
    gamma = 1- np.sum(gamma_diff,axis=1)/2
    return gamma

def negativeSlope(chroma):
    if chroma.shape[1] != 12:
        raise ValueError("invalid Chromagram shape!")
    KSPARSE = 0.038461538461538464 # normalization constant
    gamma = np.zeros(chroma.shape[0],dtype=float)
    for t in range(chroma.shape[0]):
        y = np.sort(chroma[t,:])[::-1] # sort descending
        # linear regression using numpy
        A = np.vstack([np.arange(12), np.ones((12,))]).T
        k, _ = np.linalg.lstsq(A, y, rcond=None)[0]
        #rescaled feature
        gamma[t] = 1 - np.abs(k)/KSPARSE
    return gamma

def shannonEntropy(chroma):
    """calculates the Shannon entropy of a chromagram for every timestep. The chromavector is treated as a random variable."""
    if chroma.shape[1] != 12:
        raise ValueError("invalid Chromagram shape!")
    return -np.sum(np.multiply(chroma,np.log2(chroma+np.finfo(float).eps)), axis=1)/np.log2(12)

def nonSparseness(chroma):
    if chroma.shape[1] != 12:
        raise ValueError("invalid Chromagram shape!")
    norm_l1 = np.linalg.norm(chroma, ord=1,axis=1)
    norm_l2 = np.linalg.norm(chroma, ord=2,axis=1) + np.finfo(float).eps
    gamma = 1 - ((np.sqrt(12)-norm_l1/norm_l2) / (np.sqrt(12)-1))
    return gamma

def flatness(chroma):
    if chroma.shape[1] != 12:
        raise ValueError("invalid Chromagram shape!")
    geometric_mean = np.product(chroma + np.finfo(float).eps, axis=1)**(1/12)
    arithmetic_mean = np.sum(chroma,axis=1) / 12 + np.finfo(float).eps
    return geometric_mean/arithmetic_mean

def standardDeviation(chroma):
    if chroma.shape[1] != 12:
        raise ValueError("invalid Chromagram shape!")
    return np.std(chroma,axis=1)

def angularDeviation(chroma):
    if chroma.shape[1] != 12:
        raise ValueError("invalid Chromagram shape!")
    # rearange chroma vector in fifth
    cq_fifth = np.zeros_like(chroma,dtype=float)
    for q in range(12):
        cq_fifth[:,q] = chroma[:,(q * 7) % 12]

    angles = np.exp((2*np.pi*1j*np.arange(12))/12)
    angles = np.tile(angles,(chroma.shape[0],1)) # repeat vector
    gamma = np.abs(np.sum(cq_fifth*angles,axis=1))
    return np.sqrt(1- gamma)

def intervalCategories(chroma):
    if chroma.shape[1] != 12:
        raise ValueError("invalid Chromagram shape!")
    interval_features = np.zeros((chroma.shape[0],6),dtype=float)

    for i in range(6):
        rotated_chroma = np.roll(chroma, shift=i+1, axis=1)
        interval_features[:, i] = np.sum(chroma * rotated_chroma, axis=1)
        
    return interval_features

def cqt(y,fs=22050, hop_length=1024, midi_min=12, octaves=8,bins_per_octave=36):
    estimated_tuning = librosa.estimate_tuning(y=y,sr=fs,bins_per_octave=bins_per_octave)
    cqt = np.abs(librosa.vqt(y,fmin=librosa.midi_to_hz(midi_min),
                        bins_per_octave=bins_per_octave,n_bins=bins_per_octave*octaves, sr=fs, tuning=estimated_tuning,gamma=0))
    time_vector = np.linspace(hop_length/fs,y.shape[0]/fs,cqt.shape[1])
    return time_vector, cqt

def crpChroma(y, fs=22050, nCRP=22,midinote_start=12,midinote_stop=120):
    """Chroma DCT-Reduced Log Pitch"""
    bins_per_octave = 36
    octaves = 8
    hop_length = 1024
    estimated_tuning = librosa.estimate_tuning(y=y,sr=fs,bins_per_octave=bins_per_octave)
    C = np.abs(librosa.vqt(y,fmin=librosa.midi_to_hz(midinote_start),
                        bins_per_octave=bins_per_octave,n_bins=bins_per_octave*octaves, sr=fs, tuning=estimated_tuning,gamma=0))

    # pick every third coefficient from oversampled cqt
    pitchgram_cqt = np.finfo(float).eps * np.ones((midinote_stop,C.shape[1])) 
    for note in range(midinote_start,midinote_stop):
        try:
            pitchgram_cqt[note,:] = C[(note-midinote_start)*3,:]
        except IndexError:
            break
    v = pitchgram_cqt ** 2

    vLog = np.log(100 * v + 1);  
    vLogDCT = dct(vLog, norm='ortho', axis=0);  
    vLogDCT[:nCRP,:] = 0  # liftering hochpass
    vLogDCT[nCRP,:] = 0.5 * vLogDCT[nCRP,:]

    vLog_lift = idct(vLogDCT, norm='ortho', axis=0)
    vLift = 1/100 * (np.exp(vLog_lift)-1); 
    crp = vLift.reshape(10,12,-1)
    crp = np.maximum(0, np.sum(crp, axis=0))
    crp = crp / (np.sum(crp,axis=0)+np.finfo(float).eps)
    t = np.linspace(hop_length/fs,y.shape[0]/fs,crp.shape[1])
    return t,crp.T  # transpose it so it matches the other chroma types 

if __name__ == "__main__":
    pass