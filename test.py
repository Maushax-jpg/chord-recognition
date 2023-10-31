import dataloader
import utilities
import features
import transcribe
import evaluate
import matplotlib.pyplot as plt
import numpy as np 
import scipy
import librosa.display

path = "/home/max/ET-TI/Masterarbeit/beatles/audio/03_-_All_My_Loving_instrumental.mp3"
try:
    t,sig = utilities.loadAudio(path)
    t_chroma, chroma = features.crpChroma(sig,nCRP=55)
except Exception as e:
    print(e)
    t_chroma, chroma = features.crpChroma(sig,nCRP=55)

