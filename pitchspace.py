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