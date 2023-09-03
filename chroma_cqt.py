import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
y, sr = librosa.load(librosa.ex('trumpet'))

args = {}
C = np.abs(librosa.cqt(y,**args))

fig, ax = plt.subplots()

img = librosa.display.specshow(librosa.amplitude_to_db(C, ref=np.max),

                               sr=sr, x_axis='time', ax=ax)

ax.set_title('Constant-Q power spectrum')

fig.colorbar(img, ax=ax, format="%+2.0f dB")
plt.show()