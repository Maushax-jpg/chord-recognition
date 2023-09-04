from collections import namedtuple
import numpy as np

PitchClass = namedtuple('PitchClass','name pitch_class_index chromatic_index num_accidentals')
""" 
    pitch_class_index : index of pitch class in chroma vector and list of pitch_classes
    chromatic_index : index n_c for pitch class in pitch space 
    num_accidentals : The number of accidentals n_k present in the key of this pitch class 
"""

pitch_classes = [
            PitchClass("C",0,-2,0),
            PitchClass("Db",1,-1,-5), # use enharmonic note with lowest accidentals (Db)! (C# has 7 crosses) 
            PitchClass('D',2,0,2),
            PitchClass("Eb",3,1,-3), 
            PitchClass("E",4,2,4),
            PitchClass("F",5,3,-1),
            PitchClass("F#",6,4,6),
            PitchClass("G",7,5,1),
            PitchClass("Ab",8,-6,-4), # Ab
            PitchClass("A",9,-5,3),
            PitchClass("Bb",10,-4,-2), #Bb
            PitchClass("B",11,-3,5)
]
"""A sorted list of Pitch classes: [C, C#/Db, .. , A#, B]"""

enharmonic_notes = {"C#":"Db","Db":"C#","D#":"Eb","Eb":"D#","F#":"Gb","Gb":"F#","G#":"Ab","Ab":"G#","A#":"Bb","Bb":"A#","B#":"C","C":"B#"}

def sym(n,g):
    '''SYM-operator described by Gatzsche and Mahnert'''
    try:   
        return int(np.mod((n+(g/2)),g)-(g/2))
    except ValueError:
        return None 
    
def sym3(n,g,n0):
    '''three parameter SYM-operator described by Gatzsche and Mahnert'''
    x = sym(n-n0,g)
    if x is not None:
        return x+n0
    else:
        return None

def transformChroma(chroma,alterations=False):
    """ projects the given chromagram in the pitch space
    """    
    if chroma.ndim == 1:
        chroma = np.reshape(chroma, (1, 12))
    elif chroma.shape[1] != 12:
        raise ValueError("Array shape must be (X, 12).")
    
    rho_F  = np.zeros((chroma.shape[0],),dtype=complex)
    rho_FR = np.zeros_like(chroma,dtype=complex)
    rho_TR = np.zeros_like(chroma,dtype=complex)
    rho_DR = np.zeros_like(chroma,dtype=complex)

    # iterate over all time steps
    for time_index in range(chroma.shape[0]):
        # calculte key related circles
        for x in pitch_classes:
            n_k = x.num_accidentals
            if alterations:
                n_k_sharp = pitch_classes[(x.pitch_class_index-1)%12].num_accidentals
            key_index = x.pitch_class_index
            # iterate over all chroma bins with correct index
            for pitch_class,chroma_bin in zip(pitch_classes,chroma[time_index,:]):
                n_f = sym3(49*pitch_class.chromatic_index,84,7*n_k)
                if n_k == 0:
                    # calculate vector entries for circle of fifth
                    rho_F[time_index] += chroma_bin * np.exp(-1j*2*np.pi*(n_f/84))
                # check if real pitch is part of the key n_k
                if -21 <= (n_f-7*n_k) <= 21:
                    # fifth related circle
                    n_fr = sym(n_f-7*n_k, 48)
                    rho_FR[time_index,key_index] += chroma_bin*np.exp(-1j*2*np.pi*(n_fr/48))
                    # third related circle  
                    n_tr = sym(n_f-7*n_k-12,24)
                    rho_TR[time_index,key_index] += chroma_bin*np.exp(-1j*2*np.pi*(n_tr/24))        
                    # diatonic circle   
                    n_dr = sym(n_f-7*n_k,12)
                    rho_DR[time_index,key_index] += chroma_bin*np.exp(-1j*2*np.pi*(n_dr/12))
                    continue
                if alterations:
                    # project altered notes in the circle of thirds
                    n_f_sharp = sym3(49*pitch_class.chromatic_index,84,7*n_k_sharp)
                    n_tr_sharp = sym(n_f_sharp-7*n_k-12,24)
                    rho_TR[time_index,key_index] += chroma_bin*np.exp(-1j*2*np.pi*((n_tr_sharp)/24))
    return (rho_F,rho_FR,rho_TR,rho_DR)

def getPitchClassEnergyProfile(chroma,threshold=0.6,angle_weight=0.5):
    """
    divides each chroma bin energy by the total chroma energy and applies an energy threshold afterwards.
    angle weighting puts more emphasis on tonic of a pitch class (O .. 1). 
    This is necessary because a C-Major chord is present in pitch classes C,F and G
    returns the pitch_class energies for each timestep
    """
    if chroma.ndim == 1:
        chroma = np.reshape(chroma, (1, 12))
    elif chroma.shape[1] != 12:
        raise ValueError("Array shape must be (X, 12).")
    
    angle_weighting = lambda x : -((1-angle_weight)/np.pi) * np.abs(x) + 1
    pitch_class_energy = np.zeros_like(chroma)

    chroma_energy = np.square(chroma)
    total_energy = np.expand_dims(np.sum(chroma_energy,axis=1),axis=1)

    for pitch_class in pitch_classes:
        for chroma_bin in range(12):
                # iterate all chromatic indices for every pitch class and check if pitch class is present in this key
                n_c = pitch_classes[chroma_bin].chromatic_index
                n_f = sym3(49*n_c,84,7*pitch_class.num_accidentals)
                if -21 <= (n_f-7*pitch_class.num_accidentals) <= 21:
                    n_tr = sym(n_f-7*pitch_class.num_accidentals-12,24)
                    angle = np.angle(np.exp(-1j*2*np.pi*(n_tr/24)))
                    pitch_class_energy[:,pitch_class.pitch_class_index] += angle_weighting(angle) * chroma_energy[:,chroma_bin]

    # apply tresholding for pitchclasses with low relative energy
    pitch_class_energy[pitch_class_energy < threshold * total_energy] = 0      

    pitch_class_energy = pitch_class_energy / (np.expand_dims(np.sum(pitch_class_energy,axis=1),axis=1)+np.finfo(float).eps)
    return pitch_class_energy

def filterPitchClassEnergy(pc_energy, alpha=0.95):
    if pc_energy.shape[1] != 12:
        raise ValueError("Array shape must be (X, 12).")
    
    pc_energy_filtered = np.zeros_like(pc_energy)
    pc_energy_filtered[0,:] = pc_energy[0,:] 
    for i in range(1,pc_energy.shape[0]):
        pc_energy_filtered[i,:] = alpha * pc_energy_filtered[i-1,:] + (1-alpha) * pc_energy[i,:]
    return pc_energy_filtered

def estimateKey(pc_energy):
    """estimate key by lowpass filtering of pitch class energy"""
    keys = np.argmax(pc_energy,axis=1)
    return keys