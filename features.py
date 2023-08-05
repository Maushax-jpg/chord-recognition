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
        self.loadChromaProcessor(split_nr)
        self._activations_processor = madmom.features.beats.RNNBeatProcessor()
        self._beat_processor = madmom.features.beats.BeatTrackingProcessor(fps=100)
        self._audio = None

    def saveFeatures(self,title): 
        with open(f'{os.path.join(self._basepath,"data",title)}.npy', 'wb') as f:
            np.save(f, np.array([1, 2]))  

    def loadFeatures(self,title):
        with open(f'{os.path.join(self._basepath,"data",title)}.npy', 'rb') as f:
            a = np.load(f)

    def loadAudio(self,audiopath):
        self._audio, _ = librosa.load(audiopath,sr=self._sampling_rate)

    def deepChroma(self):
        """generates a chromagram with a deep neural network"""
        return self._chroma_processor(self._audio,normalize=True)
    
    def clpChroma(self):
        """generates a chromagram with techniques that reduce overtone influence"""
        return madmom.audio.chroma.CLPChroma(self._audio,norm=True,fps=10)
    
    def crpChroma(self,nCRP=55, n=32768):
        overlap = n // 2
        midi_note_start = 12
        midi_note_stop = 120
        k = np.arange(0,n//2 + 1)
        fk = k*(self._sampling_rate/n)  # DFT-bin frequencies
        midi_note_number_frequencies = 12* np.log2((fk/440)+np.finfo(float).eps)+69

        Hp = np.zeros((fk.shape[0],midi_note_stop - midi_note_start),dtype=float)
        for i, midi_note_number in enumerate(range(12,120)):
            d = np.abs(midi_note_number - midi_note_number_frequencies)
            Hp[:,i] = 0.5 * np.tanh(np.pi * (1 - 2*d)) + 0.5

        window = np.hanning(n)
        y_blocks = buffer(self._audio,n,overlap,0,window) 
        weight = np.sum(window)
        y_spectrum = (2/weight) * np.abs(np.fft.rfft(y_blocks,axis=0))
        pitch_gram = np.matmul(Hp.T, y_spectrum) ** 2   # np.power(X,2) 

        v = np.finfo(float).eps * np.ones((120,pitch_gram.shape[1]))
        v[12:,:] = pitch_gram

        vLog = np.log(100 * v + 1);    # cf. CLP-Measure  .... diese Größe einfach zu Chroma zusammenfalten => quasi Lautheitsbewertet ...
        vLogDCT = dct(vLog, norm='ortho', axis=0);  
        vLogDCT[:nCRP,:] = 0  # liftering hochpass
        vLogDCT[nCRP,:] = 0.5 * vLogDCT[nCRP,:]

        vLog_lift = idct(vLogDCT, norm='ortho', axis=0)
        vLift = 1/100 * (np.exp(vLog_lift)-1); 
        crp = vLift.reshape(10,12,-1)
        crp = np.maximum(0, np.sum(crp, axis=0))
        crp = crp / np.sum(crp,axis=0)
        return crp


    def librosaChroma(self):
        """generates a chromagram using harmonic/percussive seperation and the CQT transform """
        y = np.copy(self._audio)
        # harmonic percussive sound separation
        y_harm = librosa.effects.harmonic(y=y, margin=8)
        # calculate chroma and transpose! shape: (t x 12)
        chroma = librosa.feature.chroma_cqt(y=y_harm, sr=self._sampling_rate,hop_length=self._hop_length).T
        return chroma
    
    def beats(self):
        # beat estimation is not very stable for short time intervals
        beat_activations = self._activations_processor(self._audio)
        return self._beat_processor(beat_activations)

    def entropy(self,chroma):
        """calculates the entropy of a chromagram for every timestep. The chromavector is treated as a random variable."""
        if chroma.shape[1] != 12:
            ax = 0
        else:
            ax = 1
        return -np.sum(np.multiply(chroma,np.log2(chroma+np.finfo(float).eps)), axis=ax)

    def loadChromaProcessor(self,split_nr):
        try:
            path = os.path.join(self._basepath,f"models/ismir2016/chroma_dnn_{split_nr}.pkl")
        except Exception as e:
            print(e)
            quit()
        self._chroma_processor = madmom.audio.chroma.DeepChromaProcessor(fmin=30, fmax=5500, unique_filters=False,models=[path])
    
    def alignChroma(self, chroma, beats, time_interval=(0,10)):
        """smooth chromagram in between beats to create a beat aligned chroma"""
        chroma_aligned = np.copy(chroma)
        # Calulate the number of frames within the time interval
        num_frames = int((time_interval[1] - time_interval[0]) * self._sampling_rate / self._hop_length)
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
    
    def normalizeChroma(self,chroma):
        """normalize chromavector so that it can be treated as a random variable"""
        max_val = np.sum(chroma, axis=1)
        return chroma / (np.expand_dims(max_val, axis=1)+np.finfo(float).eps)
    
    def rms(self,audiopath):
        """calculate the root-mean-square value of the audiosignal"""
        return madmom.audio.signal.root_mean_square(
                madmom.audio.signal.FramedSignal(audiopath, norm=True, dtype=float, fps=10)
        )     

    def selectChroma(self,chroma,time_interval=(0,10)):
        """select a time interval  of a precomputed chromagram"""
        selection = librosa.time_to_frames(time_interval,sr=self._sampling_rate,hop_length=self._hop_length)
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
