import numpy as np



def sumChromaDifferences(chroma):
    if chroma.shape[1] != 12:
        raise ValueError("invalid Chromagram shape!")
    # rearange chroma vector
    cq_fifth = np.zeros_like(chroma,dtype=float)
    for q in range(12):
        cq_fifth[:,q] = chroma[:,(q * 7) % 12]

    gamma_diff = np.zeros_like(chroma,dtype=float)
    for q in range(12):
        gamma_diff[:,q] = np.abs(cq_fifth[:,(q+1) % 12] - cq_fifth[:,q]) 
    # normalize to one
    gamma = 1- np.sum(gamma_diff,axis=1)/2
    return gamma

def negativeSlope(chroma):
    if chroma.shape[1] != 12:
        raise ValueError("invalid Chromagram shape!")
    KSPARSE = 0.038461538461538464 # normalization constant
    gamma = np.zeros(chroma.shape[0],dtype=float)
    for t in range(chroma.shape[0]):
        y = np.sort(chroma[t,:])[::-1] # sort descending
        # linear regression using numpy
        A = np.vstack([np.arange(12), np.ones((12,))]).T
        k, _ = np.linalg.lstsq(A, y, rcond=None)[0]
        #rescaled feature
        gamma[t] = 1 - np.abs(k)/KSPARSE
    return gamma

def shannonEntropy(chroma):
    """calculates the Shannon entropy of a chromagram for every timestep. The chromavector is treated as a random variable."""
    if chroma.shape[1] != 12:
        ax = 0
    else:
        ax = 1
    return -np.sum(np.multiply(chroma,np.log2(chroma+np.finfo(float).eps)), axis=ax)/np.log2(12)

def nonSparseness(chroma):
    if chroma.shape[1] != 12:
        raise ValueError("invalid Chromagram shape!")
    norm_l1 = np.linalg.norm(chroma, ord=1,axis=1)
    norm_l2 = np.linalg.norm(chroma, ord=2,axis=1) + np.finfo(float).eps
    gamma = 1 - ((np.sqrt(12)-norm_l1/norm_l2) / (np.sqrt(12)-1))
    return gamma

def flatness(chroma):
    if chroma.shape[1] != 12:
        raise ValueError("invalid Chromagram shape!")
    geometric_mean = np.product(chroma,axis=1)**(1/12)
    arithmetic_mean = np.sum(chroma,axis=1) / 12 + np.finfo(float).eps
    return geometric_mean/arithmetic_mean

def angularDeviation(chroma):
    if chroma.shape[1] != 12:
        raise ValueError("invalid Chromagram shape!")
    # rearange chroma vector in fifth
    cq_fifth = np.zeros_like(chroma,dtype=float)
    for q in range(12):
        cq_fifth[:,q] = chroma[:,(q * 7) % 12]

    angles = np.exp((2*np.pi*1j*np.arange(12))/12)
    angles = np.tile(angles,(chroma.shape[0],1)) # repeat vector
    gamma = np.abs(np.sum(cq_fifth*angles,axis=1))
    return np.sqrt(1- gamma)

def intervalCategories(chroma):
    if chroma.shape[1] != 12:
        raise ValueError("invalid Chromagram shape!")
    interval_features = np.zeros((chroma.shape[0],6),dtype=float)
    for i in range(6):
        for q in range(12):
            interval_features[:,i] += chroma[:,q] * chroma[:,(q+i+1)%12]
    return interval_features
