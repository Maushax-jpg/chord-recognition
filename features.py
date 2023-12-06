import numpy as np
import librosa
import madmom
from scipy.fftpack import dct,idct
from scipy.signal import find_peaks
from scipy.ndimage import median_filter
import itertools

def rms(y,frame_length=2048,hop_length=512):
    """compute RMS value from mono audio signal"""
    rms_lin = librosa.feature.rms(y=y,hop_length=hop_length,frame_length=frame_length).flatten()
    return np.clip(20*np.log10(rms_lin+np.finfo(float).tiny),-90,0)

def computeBeats(signal,algorithm="ellis"):
    """compute beat-activations from a madmom Signal""" 
    if algorithm == "ellis":
        tempo,beats = librosa.beat.beat_track(y=signal,units="time")
        return beats,tempo
    elif algorithm == "RNN":
        activation_processor =  madmom.features.beats.RNNBeatProcessor()
        processor = madmom.features.beats.BeatTrackingProcessor(fps=100) 
        activations = activation_processor(signal)
        beats = processor(activations)
        return beats,None
    else:
        raise ValueError("invalid algorithm!")

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

def cqt(y,fs=22050, hop_length=1024, midi_min=36, octaves=7,bins_per_octave=36):
    estimated_tuning = librosa.estimate_tuning(y=y,sr=fs,bins_per_octave=bins_per_octave)
    cqt = np.abs(librosa.vqt(y,fmin=librosa.midi_to_hz(midi_min),
                        bins_per_octave=bins_per_octave,n_bins=bins_per_octave*octaves, sr=fs, tuning=estimated_tuning,gamma=0))
    time_vector = np.linspace(hop_length/fs,y.shape[0]/fs,cqt.shape[1])
    return time_vector, cqt

def cqtChroma(sig,fs=22050,hop_length=2048,rms_thresholding=True):
    """CQT Chroma from a madmom signal
        returns time_vector (T,)
                chromagram  (T,12)
    """
    estimated_tuning = librosa.estimate_tuning(y=sig,sr=fs,bins_per_octave=36)
    chroma_cq = librosa.feature.chroma_cqt(y=sig,hop_length=hop_length, sr=fs,bins_per_octave=36,tuning=estimated_tuning)
    chroma_cq = chroma_cq / (np.sum(chroma_cq,axis=0)+np.finfo(float).eps)
    if rms_thresholding:
        rms_values = rms(sig,frame_length=4096,hop_length=hop_length)
        for i,x in enumerate(rms_values):
            # threshold chroma values if energy is below -60dB
            if x < -60:
                chroma_cq[:,i]= -1 * np.ones((12,))
    t_chroma_cq = np.linspace(sig.start,sig.stop,chroma_cq.shape[1])
    return t_chroma_cq,chroma_cq

def crpChroma(sig, fs=22050, hop_length=1024, nCRP=55,eta=100, midinote_start=12,midinote_stop=120,window=False,liftering=True,norm="l1",rms_thresholding=True):
    """Chroma DCT-Reduced Log Pitch from an madmom signal
        returns time_vector (T,)
                chromagram  (T,12)
    """
    bins_per_octave = 36
    octaves = 8
    y = np.array(sig.data)
    estimated_tuning = librosa.estimate_tuning(y=y,sr=fs,bins_per_octave=bins_per_octave)
    C = np.abs(librosa.vqt(y,fmin=librosa.midi_to_hz(midinote_start),filter_scale=1,
                        bins_per_octave=bins_per_octave,n_bins=bins_per_octave*octaves,
                        hop_length=hop_length, sr=fs, tuning=estimated_tuning,gamma=0))
    # pick every third coefficient from oversampled and tuned cqt
    pitchgram_cqt = np.finfo(float).eps * np.ones((midinote_stop,C.shape[1])) 
    
    # pitchgram window function
    for note in range(midinote_start,midinote_stop):
        try:
            if window:
                pitchgram_cqt[note,:] = np.exp(-(note-60)**2 / (2* 15**2)) * C[(note-midinote_start)*3,:]
            else:
                pitchgram_cqt[note,:] = C[(note-midinote_start)*3,:]
        except IndexError:
            break
    v = pitchgram_cqt ** 2
    
    vLog = np.log(eta * v + 1);  

    if liftering:
        vLogDCT = dct(vLog, norm='ortho', axis=0);  
        vLogDCT[:nCRP,:] = 0  # liftering hochpass
        vLogDCT[nCRP,:] = 0.5 * vLogDCT[nCRP,:]

        vLog = idct(vLogDCT, norm='ortho', axis=0)
    crp = vLog.reshape(10,12,-1)
    crp = np.sum(crp, axis=0)
    crp = np.clip(crp,0,None) # clip negative values
    if rms_thresholding:
        rms_values = rms(sig,frame_length=4096,hop_length=hop_length)
        for i,x in enumerate(rms_values):
            # threshold chroma values if energy is below -60dB
            if x < -60:
                crp[:,i]= -1 * np.ones((12,))

    if norm == "l1":
        crp = crp /np.sum(np.abs(crp)+np.finfo(float).eps,axis=0)
    elif norm == "l2":
        crp = crp / np.linalg.norm(crp)
    t = np.linspace(sig.start,sig.stop,crp.shape[1])
    return t,crp

def deepChroma(sig,split_nr=1):
    """Deep Chroma from a madmom signal
        returns time_vector (T,)
                chromagram  (T,12)
    """
    model_path = [f"/home/max/ET-TI/Masterarbeit/models/ismir2016/chroma_dnn_{split_nr}.pkl"]
    if split_nr is not None:
        dcp = madmom.audio.chroma.DeepChromaProcessor(fmin=30, fmax=5500, unique_filters=False,models=model_path)
    else:
        dcp = madmom.audio.chroma.DeepChromaProcessor()
    chroma = dcp(sig).T
    timevector = np.linspace(sig.start,sig.stop,chroma.shape[1])
    return timevector,chroma

def computeSelfSimilarityMatrix(chroma,M=1,inner=False):
    if chroma.shape[0] != 12: 
        raise ValueError(f"Invalid chromagram of shape {x.shape}")
    N = chroma.shape[1]
    x = np.zeros((M*12,N)) # preallocate matrix of embedded chromavectors
    if inner:
        for i in range(0,N-M+1):
            temp = chroma[:,i:i+M].flatten() # accumulate chromavectors
            x[:,i] =  temp / (np.linalg.norm(temp)+np.finfo(float).tiny) # normalization
        for i in range(1,M): # cosidering the last M samples
            temp = chroma[:,N-M+i:N].flatten() # accumulate chromavectors
            x[:temp.shape[0],N-M+i] =  temp / (np.linalg.norm(temp)+np.finfo(float).tiny) # normalization
        S = np.abs(np.dot(x.T,x))
    else:
        S = np.zeros((N-M+1,N-M+1),dtype=float) # self similarity matrix
        for i in range(N-M+1):
            c_1 = chroma[:,i:i+M].flatten() 
            c_1_norm = c_1 / (np.linalg.norm(c_1)+np.finfo(float).tiny)
            for col_index in range(N-M+1):
                # compute normalized embedded chromavector
                c_2 = chroma[:,col_index:col_index+M].flatten() 
                c_2_norm = c_2 / (np.linalg.norm(c_2)+np.finfo(float).tiny)
                c_diff = (c_2_norm - c_1_norm)
                S[i,col_index] = np.linalg.norm(c_diff) / 2
    return S

def smoothChromagram(chroma,W,M=1):
    """smooth chroma with weight matrix"""
    chroma_smoothed = np.zeros_like(chroma)
    for n in range(W.shape[0]):
        for m in range(M):
            temp = np.zeros((12,),dtype=float)
            if n - m < 0:  
                continue
            for i in range(W.shape[0]):
                temp += W[i,n - m] * chroma[:,i]
            chroma_smoothed[:,n] += temp / (np.sum(W[:,n - m])+ np.finfo(float).eps)
    chroma_smoothed[:,W.shape[0]:chroma.shape[1]] = chroma[:,W.shape[0]:chroma.shape[1]]
    chroma_smoothed = chroma_smoothed / np.sum(np.abs(chroma_smoothed)+np.finfo(float).eps,axis=0) # l1 normalization
    return chroma_smoothed

def computeWeightMatrix(chroma,M=15,Theta=50,inner=True):
    """Weight matrix according to  Cho and Bello: A chroma smoothing method using recurrency plots"""
    if inner: # use inner product
        S =  1 - computeSelfSimilarityMatrix(chroma,M=1,inner=True) 
        S_smoothed = 1 - computeSelfSimilarityMatrix(chroma,M=M,inner=True)
    else:
        S = computeSelfSimilarityMatrix(chroma,M=1,inner=False) 
        S_smoothed = computeSelfSimilarityMatrix(chroma,M=M,inner=False)
        
    thresholds = np.zeros((S_smoothed.shape[0],),dtype=float) # treshold for self similarity matrix
    for j in range(S_smoothed.shape[0]):
        thresholds[j] = np.sort(S_smoothed[j,:])[Theta] # the Theta most similar values are included in the threshold
    threshold_matrix = np.tile(thresholds, (S_smoothed.shape[0], 1)) # stack thresholds to a matrix

    # compute recurrence plot
    R = np.diag(np.ones(S.shape[0],))
    R[:S_smoothed.shape[0],:S_smoothed.shape[0]] = S_smoothed < threshold_matrix
    W = (1-S)*R
    return W

def applyPrefilter(sig,t_chroma,chroma,filter_type="median",**kwargs):
    if filter_type == "median":
        chroma_smoothed = median_filter(chroma, size=(1, kwargs.get("N",14)))
    elif filter_type == "beat_aligned":
        chroma_smoothed = np.zeros_like(chroma)

        beats,_ = computeBeats(sig,algorithm=kwargs.get("algorithm","ellis"))
        beat_index = []
        for beat in beats:
            beat_index.append(np.argwhere(beat <= t_chroma)[0][0])
        for i_start,i_stop in itertools.pairwise(beat_index):
            temp = np.median(chroma[:,i_start:i_stop],axis=1, keepdims=True)
            chroma_smoothed[:,i_start:i_stop] = np.broadcast_to(temp,(12,i_stop-i_start))

    elif filter_type == "rp":
        W = computeWeightMatrix(chroma,M=kwargs.get("M",25),Theta=kwargs.get("neighbors",50))
        chroma_smoothed = smoothChromagram(chroma,W,M=1)
    return chroma_smoothed



if __name__ == "__main__":
    pass