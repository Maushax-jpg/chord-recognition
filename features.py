import numpy as np
import mir_eval
import madmom
import os
import librosa
import scipy 

HOP_LENGTH = 4410
SAMPLING_RATE = 44100

class FeatureProcessor():
    def __init__(self,split_nr=0,basepath="/home/max/ET-TI/Masterarbeit/") -> None:
        self._basepath = basepath
        self._chroma_processor = self.loadChromaProcessor(split_nr)
        self._activations_processor = madmom.features.beats.RNNBeatProcessor()
        self._beat_processor = madmom.features.beats.BeatTrackingProcessor(fps=100)

    def deepChroma(self, audiopath, normalize=True, beat_alignment=True):
        """generates a chromagram with a deep neural network, uses optional beat alignment with a neural network beattracker"""
        chroma = self._chroma_processor(audiopath)
        t = np.linspace(0,chroma.shape[0]*HOP_LENGTH/SAMPLING_RATE,chroma.shape[0])    
        if beat_alignment:
            beat_activations = self._activations_processor(audiopath)
            beats = self._beat_processor(beat_activations)
            chroma = self.alignChroma(t,chroma,beats)
        if normalize:
            max_val = np.sum(chroma, axis=1)
            chroma = chroma / (np.expand_dims(max_val, axis=1)+np.finfo(float).eps)
        return t,chroma
    
    def librosaChroma(self, audiopath, normalize=True, beat_alignment=True):
        """generates a chromagram using harmonic/percussive seperation and the CQT transform """
        y,_ = librosa.load(audiopath, sr=SAMPLING_RATE)
        # harmonic percussive sound separation
        y_harm = librosa.effects.harmonic(y=y, margin=8)
        # calculate chroma
        chroma = librosa.feature.chroma_cqt(y=y_harm, sr=SAMPLING_RATE,hop_length=HOP_LENGTH).T
        t = np.linspace(0, chroma.shape[0]*HOP_LENGTH/SAMPLING_RATE, chroma.shape[0])
        if beat_alignment:
            beat_activations = self._activations_processor(audiopath)
            beats = self._beat_processor(beat_activations)
            chroma = self.alignChroma(t, chroma, beats)
        if normalize:
            max_val = np.sum(chroma, axis=1)
            chroma = chroma / (np.expand_dims(max_val, axis=1)+np.finfo(float).eps)
        return t,chroma
    
    def entropy(self,chroma):
        """calculates the entropy of a chromagram for every timestep. The chromavector is treated as a random variable."""
        return -np.sum(np.multiply(chroma,np.log2(chroma+np.finfo(float).eps)), axis=1)

    def loadChromaProcessor(self,split_nr):
        path = os.path.join(self._basepath,f"models/ismir2016/chroma_dnn_{split_nr}.pkl")
        return madmom.audio.chroma.DeepChromaProcessor(fmin=30, fmax=5500, unique_filters=False,models=[path])
    
    def alignChroma(self,t_chroma,chroma,beats):
        """smooth chromagram in between beats to create a beat aligned chroma"""
        chroma_aligned = chroma
        
        for b0,b1 in zip(beats[:-1],beats[1:]):
            # find closest index in chroma vector
            try:
                idx0 = np.argwhere(t_chroma >= b0)[0][0]
                idx1 = np.argwhere(t_chroma >= b1)[0][0]
            except IndexError:
                idx1 = t_chroma.shape[0]
            # use non-local filtering
            chroma_filter = np.minimum(chroma[idx0:idx1,:].T,
                            librosa.decompose.nn_filter(chroma[idx0:idx1,:].T,
                                                        aggregate=np.median,
                                                        metric='cosine'))
            # apply horizontal median filter
            chroma_aligned[idx0:idx1,:] = scipy.ndimage.median_filter(chroma_filter, size=(1, 9)).T
        return chroma_aligned
    
    def normalizeChroma(self,chroma):
        """normalize chromavector so that it can be treated as a random variable"""
        max_val = np.sum(chroma, axis=1)
        return chroma / (np.expand_dims(max_val, axis=1)+np.finfo(float).eps)
    
    def rms(self,audiopath):
        """calculate the root-mean-square value of the audiosignal"""
        return madmom.audio.signal.root_mean_square(
                madmom.audio.signal.FramedSignal(audiopath, norm=True, dtype=float, fps=10)
        )     
