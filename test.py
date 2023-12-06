import utilities
from nsgt import CQ_NSGT
import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

t,signal = utilities.loadAudio("/home/max/Musik/samples/letitbe_detuned.mp3",t_stop=45)

###################1
# Tuning fehlt noch
###################

class interpolate:
    def __init__(self,cqt,Ls):
        from scipy.interpolate import interp1d
        self.intp = [interp1d(np.linspace(0, Ls, len(r)), r) for r in cqt]
    def __call__(self,x):
        try:
            len(x)
        except:
            return np.array([i(x) for i in self.intp])
        else:
            return np.array([[i(xi) for i in self.intp] for xi in x])

y = signal.data
Ls = y.shape[0]  # signal length
fmin = librosa.note_to_hz("C3")
fmax = librosa.note_to_hz("C5")
n_bins = 3 * 36

# forward transform 
cq_nsgt = CQ_NSGT(fmin,fmax,n_bins,fs=22050,Ls=Ls)
cq_coeffs = np.array(cq_nsgt.forward(y),dtype=object)
cq_coeffs = map(np.abs, cq_coeffs[2:-1])
# interpolate coefficients
hop_length = 2048   # ~0.9ms
x = np.linspace(0, Ls, Ls // hop_length)
cqt = interpolate(cq_coeffs,Ls)(x)
print(cqt.shape)
fig,ax = plt.subplots()
ax.imshow(cqt)
plt.show()