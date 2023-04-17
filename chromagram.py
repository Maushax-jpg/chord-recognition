import madmom
import librosa
import numpy as np
import scipy

dcp = madmom.audio.chroma.DeepChromaProcessor()

def getChroma(filepath,chroma_type='librosa'):
    if chroma_type == 'librosa':
        y, sr = librosa.load(filepath,sr=44100)
        # harmonic percussive sound separation
        y_harm = librosa.effects.harmonic(y=y, margin=8)
        # calculate chroma
        chroma_harm = librosa.feature.chroma_cqt(y=y_harm, sr=sr,hop_length=4410)
        # use non-local filtering
        chroma_filter = np.minimum(chroma_harm,
                           librosa.decompose.nn_filter(chroma_harm,
                                                       aggregate=np.median,
                                                       metric='cosine'))
        #  apply horizontal median filter
        chroma = scipy.ndimage.median_filter(chroma_filter, size=(1, 9))
        return chroma.T
    elif chroma_type == 'madmom':
        chroma = dcp(filepath)
        return chroma
