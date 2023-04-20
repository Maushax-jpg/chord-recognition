
import dataloader
import chromagram
import joblib
import matplotlib.pyplot as plt


filepath = "/home/max/ET-TI/Masterarbeit/prototyping/data/beatles/audio/12_-_Let_It_Be/06_-_Let_It_Be.mp3"
modelpath = "/home/max/ET-TI/Masterarbeit/chord-recognition/hmm_model.pkl"

model = joblib.load(modelpath)
t,chroma = chromagram.getChroma(filepath,'madmom')
fig,ax = plt.subplots(figsize=(7,4))
chromagram.plotChroma(ax,chroma)

##########################################################

import dataloader
import chromagram
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
from matplotlib import rc
import joblib
import train_hmm
import utils

audiopath = "/home/max/ET-TI/Masterarbeit/prototyping/data/beatles/audio/12_-_Let_It_Be/06_-_Let_It_Be.mp3"
annotationspath = "/home/max/ET-TI/Masterarbeit/prototyping/data/beatles/annotations/chords/beatles_12_Let_It_Be_06_Let_It_Be.chords"
modelpath = "/home/max/ET-TI/Masterarbeit/chord-recognition/hmm_model.pkl"

# load chromagram and predict chords in given time interval
t_chroma,chroma = chromagram.getChroma(audiopath,'madmom') 
model = joblib.load(modelpath)
chord_ix_predictions = model.predict(chroma)
predictions_idx,predictions = train_hmm.postprocessing(chord_ix_predictions)

# plot waveform and annotate labels and estimated chords
fig,ax = plt.subplots()
time_interval = (0,10)
utils.plotAudioWaveform(ax,audiopath,time_interval)
utils.plotPredictionResult(ax,predictions_idx,predictions,time_interval)
utils.plotAnnotations(ax,annotationspath,time_interval)

plt.show()
