import librosa
import numpy as np
from scipy.fftpack import dct,idct
from scipy.ndimage import median_filter
import itertools

EPS = np.finfo(float).eps # machine epsilon

def rms(cqt,frame_length=2048):
    """compute a pseudo RMS value from a half-sided CQT spectrum"""
    power = 2 * np.sum(cqt, axis=1, keepdims=True) / frame_length**2
    return np.sqrt(power)

def crpChroma(y, fs=22050, hop_length=2048, nCRP=55,eta=100,window=True,liftering=True,norm="l1"):
    """compute Chroma DCT-Reduced Log Pitch feature from an audio signal"""
    bins_per_octave = 36
    octaves = 8
    midi_start = 12

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
    
    vLog = np.log(eta * v + 1);  

    if liftering:
        vLogDCT = dct(vLog, norm='ortho', axis=0);  
        vLogDCT[:nCRP,:] = 0 
        vLogDCT[nCRP,:] = 0.5 * vLogDCT[nCRP,:]
        vLog = idct(vLogDCT, norm='ortho', axis=0)

    # pitch folding
    crp = vLog.reshape(octaves,12,-1) 
    crp = np.sum(crp, axis=0)
    crp = np.clip(crp,0,None) # clip negative values

    # TODO: implement a correct RMS estimation from CQT
    rms_lin = 20*np.log10(rms(C,frame_length=2048) + EPS)
    # crp[:,energy < -60] = -1

    if norm == "l1":
        crp = crp /np.sum(np.abs(crp) + EPS, axis=0)
    elif norm == "l2":
        crp = crp / np.linalg.norm(crp)

    return crp,rms_lin

def deepChroma(y,split_nr=1,path=None):
    """compute a chromagram with the deep chroma processor"""
    import madmom
    # [f"/home/max/ET-TI/Masterarbeit/models/ismir2016/chroma_dnn_{split_nr}.pkl"]
    model_path =  path
    if split_nr is not None:
        dcp = madmom.audio.chroma.DeepChromaProcessor(fmin=30, fmax=5500, unique_filters=False,models=model_path)
    else:
        dcp = madmom.audio.chroma.DeepChromaProcessor()
    chroma = dcp(y).T
    return chroma
