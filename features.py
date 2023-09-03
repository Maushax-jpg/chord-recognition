import numpy as np
import mir_eval
import madmom
import os
import librosa
import scipy 
from utils import buffer
from scipy.fftpack import dct,idct

class FeatureProcessor():
    def __init__(self,split_nr=0,basepath="/home/max/ET-TI/Masterarbeit/",sampling_rate=44100,hop_length=4410) -> None:
        self._basepath = basepath
        self._sampling_rate = sampling_rate
        self._hop_length = hop_length
        self._activations_processor = madmom.features.beats.RNNBeatProcessor()
        self._beat_processor = madmom.features.beats.BeatTrackingProcessor(fps=100)
        self._audio = None

    def beats(self):
        # beat estimation is not very stable for short time intervals
        beat_activations = self._activations_processor(self._audio)
        return self._beat_processor(beat_activations)

    def alignChroma(self, chroma, beats, time_interval=(0,10),hop_length = None):
        """smooth chromagram in between beats to create a beat aligned chroma"""
        if hop_length is None:
            hop_length = self._hop_length
        chroma_aligned = np.copy(chroma)
        # Calulate the number of frames within the time interval
        num_frames = int((time_interval[1] - time_interval[0]) * self._sampling_rate / hop_length)
        time_vector = np.linspace(time_interval[0], time_interval[1], num_frames, endpoint=False)

        for b0,b1 in zip(beats[:-1],beats[1:]):
            # find closest index in chroma vector
            try:
                idx0 = np.argwhere(time_vector >= b0)[0][0]
            except IndexError:
                # no matching interval found at array boundaries
                idx1 = 0
            try:
                idx1 = np.argwhere(time_vector >= b1)[0][0]
            except IndexError:
                idx1 = time_vector.shape[0]  
            # use non-local filtering
            try:
                chroma_filter = np.minimum(chroma_aligned[idx0:idx1,:],
                                    librosa.decompose.nn_filter(chroma_aligned[idx0:idx1,:],
                                    axis=0,
                                    aggregate=np.median,
                                    metric='cosine'))
            except ValueError:
                chroma_filter = chroma_aligned[idx0:idx1,:]
            # apply horizontal median filter
            chroma_aligned[idx0:idx1,:] = scipy.ndimage.median_filter(chroma_filter, size=(9, 1))
        return chroma_aligned
    
    def rms(self,audiopath):
        """calculate the root-mean-square value of the audiosignal"""
        return madmom.audio.signal.root_mean_square(
                madmom.audio.signal.FramedSignal(audiopath, norm=True, dtype=float, fps=10)
        )     

    def selectChroma(self,chroma,time_interval=(0,10),hop_length=None):
        """select a time interval  of a precomputed chromagram"""
        if hop_length is None:
            hop_length = self._hop_length
        selection = librosa.time_to_frames(time_interval,sr=self._sampling_rate,hop_length=hop_length)
        idx = tuple([slice(*list(selection)),slice(None)])
        chroma_selection = np.copy(chroma[idx])
        time_vector = np.round(np.linspace(time_interval[0], time_interval[1], chroma_selection.shape[0], endpoint=False),1)
        return time_vector,chroma_selection
    
    def selectBeats(self,beats,time_interval):
        # select beats in appropriate time interval
        selected = []
        for x in beats:
            if x >= time_interval[0] and x <= time_interval[1]:
                selected.append(x)
        return np.array(selected)

def clpChroma(y,fs=44100,hop_length=4410):
        """generates a chromagram with techniques that reduce overtone influence"""
        data = madmom.audio.signal.Signal(y,fs)
        clp_processor = madmom.audio.chroma.CLPChromaProcessor(fps=fs//hop_length,norm=False)
        chroma = clp_processor(data)
        return chroma / (np.expand_dims(np.sum(chroma,axis=1), axis=1)+np.finfo(float).eps)

def sumChromaDifferences(chroma):
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
        ax = 0
    else:
        ax = 1
    return -np.sum(np.multiply(chroma,np.log2(chroma+np.finfo(float).eps)), axis=ax)/np.log2(12)

def nonSparseness(chroma):
    norm_l1 = np.linalg.norm(chroma, ord=1,axis=1)
    norm_l2 = np.linalg.norm(chroma, ord=2,axis=1) + np.finfo(float).eps
    gamma = 1 - ((np.sqrt(12)-norm_l1/norm_l2) / (np.sqrt(12)-1))
    return gamma

def flatness(chroma):
    geometric_mean = np.product(chroma,axis=1)**(1/12)
    arithmetic_mean = np.sum(chroma,axis=1) / 12 + np.finfo(float).eps
    return geometric_mean/arithmetic_mean

def angularDeviation(chroma):
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
        for q in range(12):
            interval_features[:,i] += chroma[:,q] * chroma[:,(q+i+1)%12]
    return interval_features

def hpss(y,margin=1):
    """Harmonic Percussive Source Separation  (Fitzgerald)
    based on Fitzgerald, Derry, Driedger, Müller and Disch"""
    return librosa.effects.harmonic(y=y, margin=margin)

def crpChroma(y,nCRP=22, n=2**14,overlap=int(2**14 *0.75),cqt=True):
    midi_note_start = 12
    midi_note_stop = 120

    if cqt: # use CQT for the pitchgram 
        bins_per_octave = 36
        octaves = 8
        estimated_tuning = librosa.estimate_tuning(y=y,bins_per_octave=bins_per_octave)
        C = np.abs(librosa.vqt(y,fmin=librosa.midi_to_hz(midi_note_start),
                            bins_per_octave=bins_per_octave,n_bins=bins_per_octave*octaves, tuning=estimated_tuning,gamma=0))

        # pick every third coefficient from oversampled cqt
        pitchgram_cqt = np.finfo(float).eps * np.ones((midi_note_stop,C.shape[1])) 
        for note in range(midi_note_start,midi_note_stop):
            try:
                pitchgram_cqt[note,:] = C[(note-midi_note_start)*3,:]
            except IndexError:
                break
        v = pitchgram_cqt ** 2
    else: # use STFT with filterbank
        k = np.arange(0,n//2 + 1)
        fk = k*(fs/n)  # DFT-bin frequencies
        midi_note_number_frequencies = 12* np.log2((fk/440)+np.finfo(float).eps)+69

        Hp = np.zeros((fk.shape[0],midi_note_stop - midi_note_start),dtype=float)
        for i, midi_note_number in enumerate(range(midi_note_start,midi_note_stop)):
            d = np.abs(midi_note_number - midi_note_number_frequencies)
            Hp[:,i] = 0.5 * np.tanh(np.pi * (1 - 2*d)) + 0.5

        window = np.hanning(n)
        y_blocks = buffer(y,n,overlap,0,window) 
        weight = np.sum(window)
        y_spectrum = (2/weight) * np.abs(np.fft.rfft(y_blocks,axis=0))
        pitch_gram = np.matmul(Hp.T, y_spectrum) ** 2   # np.power(X,2) 

        v = np.finfo(float).eps * np.ones((midi_note_stop,pitch_gram.shape[1]))
        v[12:,:] = pitch_gram

    vLog = np.log(100 * v + 1);  
    vLogDCT = dct(vLog, norm='ortho', axis=0);  
    vLogDCT[:nCRP,:] = 0  # liftering hochpass
    vLogDCT[nCRP,:] = 0.5 * vLogDCT[nCRP,:]

    vLog_lift = idct(vLogDCT, norm='ortho', axis=0)
    vLift = 1/100 * (np.exp(vLog_lift)-1); 
    crp = vLift.reshape(10,12,-1)
    crp = np.maximum(0, np.sum(crp, axis=0))
    crp = crp / (np.sum(crp,axis=0)+np.finfo(float).eps)
    return crp.T  # transpose it so it matches the other chroma types 

def deepChroma(y,split_nr=1):
    path = f"/home/max/ET-TI/Masterarbeit/models/ismir2016/chroma_dnn_{split_nr}.pkl"
    chroma_processor = madmom.audio.chroma.DeepChromaProcessor(fmin=30, fmax=5500, unique_filters=False,models=[path])
    chroma = chroma_processor(y)
    chroma = chroma / np.expand_dims(np.sum(chroma,axis=1) + np.finfo(float).eps,axis=1)
    return chroma

def librosaChroma(y,fs=44100,hop_length=4096,hpss=True):
    """generates a chromagram using harmonic/percussive seperation and the CQT transform """
    # harmonic percussive sound separation
    if hpss:
        y_harm = librosa.effects.harmonic(y=y, margin=8)
    else:
        y_harm = y
    # calculate chroma and transpose! shape: (t x 12)
    chroma = librosa.feature.chroma_cqt(y=y_harm, sr=fs,hop_length=hop_length).T
    return chroma / (np.expand_dims(np.sum(chroma,axis=1), axis=1)+np.finfo(float).eps)

def RNN_beats(filepath):
    activations_processor = madmom.features.beats.RNNBeatProcessor(online=True)
    beat_processor = madmom.features.beats.BeatTrackingProcessor(fps=100)
    return beat_processor(activations_processor(filepath))

def alignChroma(chroma, beats, time_interval=(0,10),hop_length = 4410,sr=44100,filtering=False):
    """smooth chromagram in between beats to create a beat aligned chroma"""
    chroma_aligned = np.copy(chroma)
    # Calulate the number of frames within the time interval
    num_frames = int((time_interval[1] - time_interval[0]) * sr / hop_length)
    time_vector = np.linspace(time_interval[0], time_interval[1], num_frames, endpoint=False)

    for b0,b1 in zip(beats[:-1],beats[1:]):
        # find closest index in chroma vector
        try:
            idx0 = np.argwhere(time_vector >= b0)[0][0]
        except IndexError:
            # no matching interval found at array boundaries
            idx1 = 0
        try:
            idx1 = np.argwhere(time_vector >= b1)[0][0]
        except IndexError:
            idx1 = time_vector.shape[0]  
        # use non-local filtering
        if filtering:
            try:
                chroma_filter = np.minimum(chroma_aligned[idx0:idx1,:],
                                    librosa.decompose.nn_filter(chroma_aligned[idx0:idx1,:],
                                    axis=0,
                                    aggregate=np.median,
                                    metric='cosine'))
            except ValueError:
                chroma_filter = chroma_aligned[idx0:idx1,:]
        else:
            chroma_filter = chroma_aligned[idx0:idx1,:]
        # apply horizontal median filter
        chroma_aligned[idx0:idx1,:] = scipy.ndimage.median_filter(chroma_filter, size=(9, 1))
    return chroma_aligned
