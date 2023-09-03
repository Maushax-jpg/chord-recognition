import matplotlib.pyplot as plt
import numpy as np
import librosa.display
import librosa
from scipy.fftpack import dct,idct

path = "/home/max/ET-TI/Masterarbeit/mirdata/beatles/"
title = "06_-_Rubber_Soul/11_-_In_My_Life"
title = "12_-_Let_It_Be/06_-_Let_It_Be"
bins_per_octave = 12
octaves = 7
tstop=5
y, sr = librosa.load(path+"audio/"+title+".wav",sr=44100,duration=tstop)
C = np.abs(librosa.vqt(y, sr=sr,fmin=librosa.note_to_hz("C2"),
                       bins_per_octave=bins_per_octave,n_bins=bins_per_octave*octaves,
                         hop_length=2048,
                         gamma=0))
nCRP = 25
n = bins_per_octave*octaves+12
v = np.finfo(float).eps * np.ones((bins_per_octave*octaves+12,C.shape[1]))
v[12:,:] = C
vLog = np.log(100 * v + 1);    # cf. CLP-Measure  .... diese Größe einfach zu Chroma zusammenfalten => quasi Lautheitsbewertet ..
vLogDCT = dct(vLog, norm='ortho', axis=0);  
vLogDCT[:nCRP,:] = 0  # liftering hochpass
vLogDCT[nCRP,:] = 0.5 * vLogDCT[nCRP,:]

vLog_lift = idct(vLogDCT, norm='ortho', axis=0)
vLift = 1/100 * (np.exp(vLog_lift)-1); 
crp_cqt = vLift.reshape(n//12,12,-1)

crp_cqt = np.maximum(0, np.sum(crp_cqt, axis=0))
crp_cqt = crp_cqt / np.sum(crp_cqt,axis=0)
crp_cqt = crp_cqt.T
time_vector = np.linspace(0, tstop, crp_cqt.shape[0], endpoint=False)

fig, ax = plt.subplots()
img = librosa.display.specshow(crp_cqt.T,x_coords=time_vector.T,x_axis='time', y_axis='chroma', cmap="Reds", ax=ax, vmin=0, vmax=0.5)
ax.text(0,12,"CRP")
plt.show()

quit()
img = librosa.display.specshow(C, y_axis='cqt_note',bins_per_octave=bins_per_octave, cmap="Reds", ax=ax, vmin=0, vmax=0.5)
plt.show()
