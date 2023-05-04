import os
import re
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset
import mir_eval
import mirdata
import madmom
import librosa
import librosa.display
import numpy as np
import scipy
import audioread.ffdec


SAMPLING_RATE = 44100
HOP_LENGTH = 4410

class MIRDataset(Dataset):
    """
    Dataloader class for MIR Datasets. Available Datasets are "guitarset","beatles" and "queen"
    """
    def __init__(self,name,basepath="/home/max/ET-TI/Masterarbeit/mirdata/",use_deep_chroma=True,align_chroma=True):
        super().__init__()
        self._path = basepath
        self._use_deep_chroma = use_deep_chroma
        self._align_chroma = align_chroma
        self._dataset = self.loadDataset(name)
        self._tracks = self._dataset.load_tracks()
        self._chroma_processor = madmom.audio.chroma.DeepChromaProcessor()
        self._activations_processor = madmom.features.beats.RNNBeatProcessor()
        self._beat_processor = madmom.features.beats.BeatTrackingProcessor(fps=100)


    def __len__(self):
        """returns rows of dataframe"""
        return len(self._dataset.track_ids)
    
    def __getitem__(self, track_id):
        print(f"LOADING AUDIO: \n{self._tracks[track_id].title} ID: {track_id}")
        t,chroma = self.calculateChroma(track_id)
        ref_intervals,ref_labels = self.getAnnotations(track_id)
        audio,_ = self._tracks[track_id].audio
        return audio,t,chroma,ref_intervals,ref_labels

    def getTrackIDs(self):
        """returns list of available track IDs that can be used to access the dataset tracks"""
        return self._dataset.track_ids
    
    def loadDataset(self,name):
        if name != "beatles":
            raise ValueError(f"Dataset {name} not available!")
        return mirdata.initialize(name, data_home=os.path.join(self._path,name))
    
    def calculateChroma(self,track_id):
        """
        calculate Chromagram with madmom's chroma processor or librosa.
        optionally align the chromagram with madom's beat tracker.
        return timevector t, chromagram chroma
        """
        if self._use_deep_chroma:
            chroma = self._chroma_processor(self._tracks[track_id].audio_path)
            t = np.linspace(0,chroma.shape[0]*HOP_LENGTH/SAMPLING_RATE,chroma.shape[0])    
        else:
            y,sr = self._tracks[track_id].audio
            if sr != SAMPLING_RATE:
                raise ValueError(f"Invalid Sampling rate used by mirdata! sr = {sr}")
            # harmonic percussive sound separation
            y_harm = librosa.effects.harmonic(y=y, margin=8)
            # calculate chroma
            chroma = librosa.feature.chroma_cqt(y=y_harm, sr=SAMPLING_RATE,hop_length=HOP_LENGTH).T
            t = np.linspace(0,chroma.shape[0]*HOP_LENGTH/SAMPLING_RATE,chroma.shape[0])
        
        if self._align_chroma:
            beat_activations = self._activations_processor(self._tracks[track_id].audio_path)
            beats = self._beat_processor(beat_activations)
            chroma = self.alignChroma(t,chroma,beats)
        return t,chroma

    def alignChroma(self,t_chroma,chroma,beats):
        """beat align chroma"""
        chroma_aligned = chroma
        
        for b0,b1 in zip(beats[:-1],beats[1:]):
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
    
    def getAnnotations(self,track_id):
        ref_intervals,ref_labels = mir_eval.io.load_labeled_intervals(self._tracks[track_id].chords_path,' ','#')
        return ref_intervals,ref_labels
