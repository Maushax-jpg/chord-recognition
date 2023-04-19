import madmom
import librosa
import librosa.display
import numpy as np
import scipy

dcp = madmom.audio.chroma.DeepChromaProcessor()
SAMPLING_RATE = 44100
HOP_LENGTH = 4410

def plotChroma(ax,chroma,sr=SAMPLING_RATE,hop_length=HOP_LENGTH):
    
    librosa.display.specshow(chroma[:100,:].T,y_axis='chroma',cmap='viridis',
                                ax=ax,x_axis='time',sr=sr,hop_length=hop_length)
    ax.set_xlabel('Time in s')

def getChroma(filepath,chroma_type='librosa'):
    if chroma_type == 'librosa':
        y, sr = librosa.load(filepath,sr=44100)
        # harmonic percussive sound separation
        y_harm = librosa.effects.harmonic(y=y, margin=8)
        # calculate chroma
        chroma_harm = librosa.feature.chroma_cqt(y=y_harm, sr=SAMPLING_RATE,hop_length=HOP_LENGTH)
        # use non-local filtering
        chroma_filter = np.minimum(chroma_harm,
                           librosa.decompose.nn_filter(chroma_harm,
                                                       aggregate=np.median,
                                                       metric='cosine'))
        #  apply horizontal median filter
        chroma = scipy.ndimage.median_filter(chroma_filter, size=(1, 9))
        t = np.linspace(HOP_LENGTH/SAMPLING_RATE,chroma.shape[0]*HOP_LENGTH/SAMPLING_RATE,chroma.shape[0])
        return t,chroma.T
    elif chroma_type == 'madmom':
        chroma = dcp(filepath)
        t = np.linspace(HOP_LENGTH/SAMPLING_RATE,chroma.shape[0]*HOP_LENGTH/SAMPLING_RATE,chroma.shape[0])
        return t,chroma
