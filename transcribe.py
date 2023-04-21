import dataloader
import chromagram
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
from matplotlib import rc
import joblib
import train_hmm
import utils

if __name__ == '__main__':
    audiopath = "/home/max/ET-TI/Masterarbeit/prototyping/data/beatles/audio/12_-_Let_It_Be/06_-_Let_It_Be.mp3"
    annotationspath = "/home/max/ET-TI/Masterarbeit/prototyping/data/beatles/annotations/chords/beatles_12_Let_It_Be_06_Let_It_Be.chords"
    modelpath = "/home/max/ET-TI/Masterarbeit/chord-recognition/hmm_model.pkl"

    # load chromagram and predict chords in given time interval
    t_chroma,chroma = chromagram.getChroma(audiopath,'madmom') 
    model = joblib.load(modelpath)

    x = model.predict(chroma)

