import numpy as np
from scipy.fftpack import dct, idct
import matplotlib.pyplot as plt
# Generate random data (size 8x10)
x = np.arange(0,36)

y = np.ones((36,3))
y[:,0] = x
y[:,1] = x
data = y.reshape(3,12,3)
print(data[0,:,0])




