import numpy as np
from scipy.fftpack import dct, idct
import matplotlib.pyplot as plt
import features
import librosa
fs = 44100
y,_ = librosa.load("/home/max/ET-TI/Masterarbeit/mirdata/beatles/audio/12_-_Let_It_Be/06_-_Let_It_Be.wav",sr = fs)
featureprocessor = features.FeatureProcessor(split_nr=1)

n = int(6*8192)
print(n//2 / fs)
#featureprocessor.crpChroma(55)




