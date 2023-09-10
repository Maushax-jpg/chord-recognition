import utilities
import numpy as np
import librosa
import features
import matplotlib.pyplot as plt
import audioread
import madmom

start = 0
stop = 30

# original audio
# path = "/home/max/ET-TI/Masterarbeit/mirdata/beatles/audio/10CD1_-_The_Beatles/CD1_-_17_-_Julia.wav"
path = "/home/max/ET-TI/Masterarbeit/mirdata/beatles/audio/12_-_Let_It_Be/06_-_Let_It_Be.wav"
try:
    y,sr = librosa.load(path,mono=True,offset=start,duration=stop-start,sr=22050)
except TypeError as e: 
    print(e)
    y,sr = librosa.load(path,mono=True,offset=start,duration=stop-start,sr=22050)
t,chroma = features.crpChroma(y,nCRP=30)

# audio with no vocals
# path = "/home/max/Musik/beatles_julia.mpga"
path = "/home/max/Musik/beatles_letitbe.mpga"
with audioread.audio_open(path) as f:
    y,sr = librosa.load(f,mono=True,offset=start,duration=stop-start,sr=22050)

# beat_processor = madmom.features.downbeats.DBNDownBeatTrackingProcessor(2,fps=10)
# activation_processor =  madmom.features.downbeats.RNNDownBeatProcessor()
# activations = activation_processor(path)
t,chroma_instrumental = features.crpChroma(y,nCRP=30)
print(chroma_instrumental.shape)
fig,ax = plt.subplots(2,1,height_ratios=(5,5),figsize=(9,5))
utilities.plotChromagram(ax[0],t,chroma)
# pitch class energy can conveniently be visualized with the plotChromagram function
utilities.plotChromagram(ax[1],t,chroma_instrumental)
fig.savefig("/home/max/Downloads/test.png")