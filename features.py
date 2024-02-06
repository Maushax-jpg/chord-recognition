import librosa
import numpy as np
from scipy.fftpack import dct,idct
from scipy.ndimage import median_filter
from scipy.stats import pearsonr
import utils
import os
import madmom

EPS = np.finfo(float).eps # machine epsilon

def sumChromaDifferences(chroma):
    if chroma.shape[0] != 12:
        raise ValueError("invalid Chromagram shape!")
    # rearange chroma vector
    cq_fifth = np.zeros_like(chroma,dtype=float)
    for q in range(12):
        cq_fifth[q,:] = chroma[(q * 7) % 12, :]

    gamma_diff = np.zeros_like(chroma,dtype=float)
    for q in range(12):
        gamma_diff[q, :] = np.abs(cq_fifth[(q+1) % 12, :] - cq_fifth[q,:]) 
    # normalize to one
    gamma = 1- np.sum(gamma_diff,axis=0)/2
    return gamma

def negativeSlope(chroma):
    if chroma.shape[0] != 12:
        raise ValueError("invalid Chromagram shape!")
    KSPARSE = 0.038461538461538464 # normalization constant
    gamma = np.zeros(chroma.shape[1],dtype=float)
    for t in range(chroma.shape[1]):
        y = np.sort(chroma[:, t])[::-1] # sort descending
        # linear regression using numpy
        A = np.vstack([np.arange(12), np.ones((12,))]).T
        k, _ = np.linalg.lstsq(A, y, rcond=None)[0]
        #rescaled feature
        gamma[t] = 1 - np.abs(k)/KSPARSE
    return gamma

def shannonEntropy(chroma):
    """calculates the Shannon entropy of a chromagram for every timestep. The chromavector is treated as a random variable."""
    if chroma.shape[0] != 12:
        raise ValueError("invalid Chromagram shape!")
    return -np.sum(np.multiply(chroma,np.log2(chroma+np.finfo(float).eps)), axis=0)/np.log2(12)

def nonSparseness(chroma):
    if chroma.shape[0] != 12:
        raise ValueError("invalid Chromagram shape!")
    norm_l1 = np.linalg.norm(chroma, ord=1,axis=0)
    norm_l2 = np.linalg.norm(chroma, ord=2,axis=0) + np.finfo(float).eps
    gamma = 1 - ((np.sqrt(12)-norm_l1/norm_l2) / (np.sqrt(12)-1))
    return gamma

def flatness(chroma):
    if chroma.shape[0] != 12:
        raise ValueError("invalid Chromagram shape!")
    geometric_mean = np.product(chroma + np.finfo(float).eps, axis=0)**(1/12)
    arithmetic_mean = np.sum(chroma,axis=0) / 12 + np.finfo(float).eps
    return geometric_mean/arithmetic_mean

def computeRMS(y, fs=22050, hop_length=2048):
    S,_ = librosa.magphase(librosa.stft(y, n_fft=4096, hop_length=hop_length))
    rms = librosa.feature.rms(S=S, frame_length=4096, hop_length=hop_length)[0]
    rms = 20 * np.log10(rms + EPS)
    np.clip(rms, -80, 0, out=rms)
    return rms

def crpChroma(y, fs=22050, hop_length=2048, nCRP=33,eta=100,window=False,compression=True,liftering=True,norm="l1",clip=True):
    """compute Chroma DCT-Reduced Log Pitch feature from an audio signal"""
    bins_per_octave = 36
    octaves = 8
    midi_start = 24

    estimated_tuning = librosa.estimate_tuning(y=y,sr=fs,bins_per_octave=bins_per_octave)
    C = np.abs(librosa.vqt(y,fmin=librosa.midi_to_hz(12),filter_scale=1,
                        bins_per_octave=bins_per_octave,n_bins=bins_per_octave*octaves,
                        hop_length=hop_length, sr=fs, tuning=estimated_tuning,gamma=0))
    
    # pick every third coefficient from oversampled and tuned cqt
    pitchgram_cqt = EPS * np.ones((octaves*12,C.shape[1])) 
    
    # pitchgram window function
    for note in range(midi_start,midi_start+12*octaves):
        if window:
            # weight and pick every third amplitude value
            pitchgram_cqt[note - midi_start,:] = np.exp(-(note-60)**2 / (2* 15**2)) * C[(note-midi_start)*3,:] 
        else:
            # no weighting
            pitchgram_cqt[note - midi_start,:] = C[(note-midi_start)*3,:] 

    v = pitchgram_cqt ** 2
    if compression:
        v = np.log(eta * v + 1);  

    if liftering:
        vLogDCT = dct(v, norm='ortho', axis=0);  
        vLogDCT[:nCRP,:] = 0 
        vLogDCT[nCRP,:] = 0.5 * vLogDCT[nCRP,:]
        v = idct(vLogDCT, norm='ortho', axis=0)

    # pitch folding
    crp = v.reshape(octaves,12,-1) 
    crp = np.sum(crp, axis=0)

    if clip:
        crp = np.clip(crp,0,None) # clip negative values

    if norm == "l1":
        crp = crp /np.sum(np.abs(crp) + EPS, axis=0)
    elif norm == "l2":
        crp = crp / np.linalg.norm(crp)
    return crp

def deepChroma(filepath,split_nr=1):
    """compute a chromagram with the deep chroma processor"""
    model_path =  os.path.join(os.path.split(__file__)[0],f"models/chroma_dnn_{split_nr}.pkl")
    if split_nr is not None:
        dcp = madmom.audio.chroma.DeepChromaProcessor(fmin=30, fmax=5500, unique_filters=False,models=[model_path])
    else:
        dcp = madmom.audio.chroma.DeepChromaProcessor()
    chroma = dcp(filepath).T
    return chroma

def computeSSM(chroma,M=1):
    """computation of a self-similarity matrix from a chromagram with (optional) temporal embedding.
        Zero-padding is applied at the end of the chroma sequence"""
    if chroma.shape[0] != 12: 
        raise ValueError(f"Invalid chromagram of shape {chroma.shape}")
    N = chroma.shape[1]
    x = np.zeros((M*12,N)) # preallocate matrix of embedded chromavectors
    for i in range(0,N-M+1):
        temp = chroma[:,i:i+M].flatten() # accumulate chromavectors
        x[:,i] =  temp / (np.linalg.norm(temp)+EPS) # normalization
    for i in range(1,M): # cosidering the last M samples
        temp = chroma[:,N-M+i:N].flatten() # accumulate chromavectors
        x[:temp.shape[0],N-M+i] =  temp / (np.linalg.norm(temp)+EPS) # normalization
    SSM = np.abs(np.dot(x.T,x)) # compute a self-similarity matrix
    return SSM

def computeWeightMatrix(chroma,M=15,neighbors=50):
    """compute a weight matrix for finding similarities in the chroma
      see:  Cho and Bello: A chroma smoothing method using recurrency plots"""
    ssm =  computeSSM(chroma,M=1) 
    ssm_smoothed = computeSSM(chroma,M=M)
    N = chroma.shape[1]
    thresholds = np.zeros((N,),dtype=float) # treshold for self similarity matrix
    for j in range(N):
        thresholds[j] = np.sort(ssm_smoothed[j,:])[-neighbors] # the most similar values are included in the threshold
    threshold_matrix = np.tile(thresholds, (N, 1)) # stack thresholds to a matrix

    # compute recurrence plot
    recurrence_plot = ssm_smoothed > threshold_matrix
    weight_matrix = ssm*recurrence_plot
    return weight_matrix,ssm,ssm_smoothed

def computeCorrelation(chroma,inner_product=True,template_type="majmin"):
    templates,labels = utils.createChordTemplates(template_type=template_type) 
    if inner_product:
        correlation = np.matmul(templates.T,chroma)
        np.clip(correlation,out=correlation,a_min=0,a_max=1)
    else:
        correlation = np.zeros((templates.shape[1],chroma.shape[1]),dtype=float)
        # using scipy pearsons correlation coefficient
        for i in range(templates.shape[1]):
            for t in range(chroma.shape[1]):
                correlation[i,t] = pearsonr(templates[:,i],chroma[:,t]).statistic

        # replace NaN with zeros
        correlation = np.nan_to_num(correlation)
        np.clip(correlation,out=correlation,a_min=0,a_max=1)
    return correlation,labels

def applyPrefilter(t_chroma, chroma, filter_type,**params):
    """
    smooth chromagram with median or recurrence plot
    @params: templates
    filter_type = "median" \\
    N .. median filter length, defaults to 14

    filter_type = "rp" \\
    M .. temporal embedding of self-similarity matrix , default = 25\\
    neighbors .. number of chroma vectors included in the adaptive threshold for computing the recurrence plot , default = 50 
    """
    chroma_smoothed = np.zeros_like(chroma)
    if filter_type == "median":
        N = params.get("prefilter_length",14)
        chroma_smoothed = median_filter(chroma, size=(1, N))
    elif filter_type == "rp":
        M = params.get("embedding",25)
        THETA = params.get("neighbors",50)
        W,SSM,SSM_M = computeWeightMatrix(chroma,M=M,neighbors=THETA)
        if params.get("display",False):
            utils.plotRecurrencePlots(t_chroma,W,SSM,SSM_M)
        chroma_smoothed = np.zeros_like(chroma)
        for i in range(W.shape[0]):
            temp = np.zeros((12,),dtype=float)
            for j in range(W.shape[0]):
                temp += W[i,j] * chroma[:,j]
            chroma_smoothed[:,i] = temp / (np.sum(W[:,j])+ EPS)
        chroma_smoothed = chroma_smoothed / np.sum(np.abs(chroma_smoothed)+ EPS,axis=0) # l1 normalization
    else:
        raise NotImplementedError(f"filter type {filter_type} not implemented")
    return chroma_smoothed

def uniform_transition_matrix(p=0.01, N=24):
    """Computes uniform transition matrix
    source: https://www.audiolabs-erlangen.de/resources/MIR/FMP/C5/C5S3_ChordRec_HMM.html
    Notebook: C5/C5S3_ChordRec_HMM.ipynb

    Args:
        p (float): Self transition probability (Default value = 0.01)
        N (int): Column and row dimension (Default value = 24)

    Returns:
        A (np.ndarray): Output transition matrix
    """
    off_diag_entries = (1-p) / (N-1)     # rows should sum up to 1
    A = off_diag_entries * np.ones([N, N])
    np.fill_diagonal(A, p)
    return A

def viterbi_log_likelihood(A, C, B_O):
    """Viterbi algorithm (log variant) for solving the uncovering problem
    source: https://www.audiolabs-erlangen.de/resources/MIR/FMP/C5/C5S3_ChordRec_HMM.html
    Notebook: C5/C5S3_Viterbi.ipynb

    Args:
        A (np.ndarray): State transition probability matrix of dimension I x I
        C (np.ndarray): Initial state distribution  of dimension I
        B_O (np.ndarray): Likelihood matrix of dimension I x N

    Returns:
        S_opt (np.ndarray): Optimal state sequence of length N
        S_mat (np.ndarray): Binary matrix representation of optimal state sequence
        D_log (np.ndarray): Accumulated log probability matrix
        E (np.ndarray): Backtracking matrix
    """
    I = A.shape[0]    # Number of states
    N = B_O.shape[1]  # Length of observation sequence
    tiny = np.finfo(float).tiny
    A_log = np.log(A + tiny)
    C_log = np.log(C + tiny)
    B_O_log = np.log(B_O + tiny)

    # Initialize D and E matrices
    D_log = np.zeros((I, N))
    E = np.zeros((I, N-1)).astype(np.int32)
    D_log[:, 0] = C_log + B_O_log[:, 0]

    # Compute D and E in a nested loop
    for n in range(1, N):
        for i in range(I):
            temp_sum = A_log[:, i] + D_log[:, n-1]
            D_log[i, n] = np.max(temp_sum) + B_O_log[i, n]
            E[i, n-1] = np.argmax(temp_sum)

    # Backtracking
    S_opt = np.zeros(N).astype(np.int32)
    S_opt[-1] = np.argmax(D_log[:, -1])
    for n in range(N-2, -1, -1):
        S_opt[n] = E[int(S_opt[n+1]), n]

    # Matrix representation of result
    S_mat = np.zeros((I, N)).astype(np.int32)
    for n in range(N):
        S_mat[S_opt[n], n] = 1

    return S_mat, S_opt, D_log, E

def applyPostfilter(correlation,labels,filter_type,**params):
    """apply a postfilter to the correlation matrix"""
    if filter_type == "median":
        N = params.get("postfilter_length",4)
        correlation_smoothed = median_filter(correlation, size=(1, N))
    elif filter_type == "hmm":
        p = params.get("transition_prob",0.1)
        A = uniform_transition_matrix(p,len(labels)) # state transition probability matrix
        B_O = correlation / (np.sum(correlation,axis=0) + EPS) # likelyhood matrix -> quasi normalized inner product
        C = np.ones((len(labels,))) * 1/len(labels)   # initial state probability matrix
        correlation_smoothed, _, _, _ = viterbi_log_likelihood(A, C, B_O)
    return correlation_smoothed


