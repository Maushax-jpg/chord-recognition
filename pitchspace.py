from collections import namedtuple
import numpy as np
import mir_eval
import matplotlib
import utilities
import features

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


def angle_between(z1,z2):
    """compute shortest angle difference between two complex numbers"""
    diff = np.abs(np.angle(z1) - np.angle(z2))
    if diff > np.pi:
        diff = 2*np.pi - diff
    return diff


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
        # iterate all chromatic indices for every pitch class and check if pitch class is present in this key
        r_tr = 0+0j
        for chroma_bin in range(12):
                n_c = pitch_classes[chroma_bin].chromatic_index
                n_f = sym3(49*n_c,84,7*pitch_class.num_accidentals)
                if -21 <= (n_f-7*pitch_class.num_accidentals) <= 21:
                    n_tr = sym(n_f-7*pitch_class.num_accidentals-12,24)
                    r_tr += np.exp(-1j*2*np.pi*(n_tr/24))
                    # accumulate chroma energy
                    pitch_class_energy[:,pitch_class.pitch_class_index] += chroma_energy[:,chroma_bin]
        # apply angle weighting
        pitch_class_energy[:,pitch_class.pitch_class_index] = pitch_class_energy[:,pitch_class.pitch_class_index] * angle_weighting(np.angle(r_tr))
    # apply tresholding for pitchclasses with low relative energy
    pitch_class_energy[pitch_class_energy < threshold * total_energy] = 0      

    pitch_class_energy = pitch_class_energy / (np.expand_dims(np.sum(pitch_class_energy,axis=1),axis=1)+np.finfo(float).eps)
    return pitch_class_energy

def filterPitchClassEnergy(pc_energy,fifth_width=None,rms=None, alpha=0.95):
    if pc_energy.shape[1] != 12:
        raise ValueError("Array shape must be (X, 12).")
    if fifth_width is not None and rms is not None:
        # dampen values with a high fifth width
        alpha_vector = np.ones_like(fifth_width) * 0.99
        index = fifth_width > 0.66
        if index.any():
            alpha_vector[index] = 1
        index = rms < 0.01
        if index.any():
            alpha_vector[index] = 0
    pc_energy_filtered = np.zeros_like(pc_energy)
    pc_energy_filtered[0,:] = pc_energy[0,:] 
    for i in range(1,pc_energy.shape[0]):
        pc_energy_filtered[i,:] = alpha_vector[i] * pc_energy_filtered[i-1,:] + (1-alpha_vector[i]) * pc_energy[i,:]

    return pc_energy_filtered

def estimateKeys(chroma,ic_threshold=0.01):
    # other approach to calculate correlation with major key profiles (krumhansl)
    templates = np.zeros((12,12),dtype=float)
    key_profile = np.array([5, 2, 3.5, 2, 4.5, 4, 2, 4.5, 2, 3.5, 1.5, 4])/12 # (12,) arbitrary normation for plots
    # key_profile = key_profile / np.sum(key_profile)
    for i in range(12):
        templates[i,:] = np.roll(key_profile,i)
    correlation = np.matmul(templates,chroma.T).T # chroma shape (t,12)

    # computation of interval categories
    key_profile = [0,2,4,5,7,9,11]
    key_template = np.zeros_like(chroma)
    key_template[:, key_profile] = 1.0
    # precompute interval_categories for all keys
    interval_categories = np.zeros((12,chroma.shape[0],6))
    ic_energy = np.zeros_like(chroma)
    for pitch_class in range(12):
        key_related_chroma = np.multiply(chroma,np.roll(key_template,pitch_class))
        interval_categories[pitch_class,:,:] = features.intervalCategories(key_related_chroma)
        ic_energy[:,pitch_class] = np.sum(interval_categories[pitch_class,:,2:5], axis=1)  # sum energy of m3/M6,M3/m6,P4/P5  

    # we can savely discard some keys with low correlation (6 or 8?)
    correlation_energy = np.square(correlation)
    key_candidates = np.argsort(correlation_energy,axis=1)[:,-3:]

    keys = []
    for i in range(key_candidates.shape[0]): # for all timesteps
        candidates = []
        for n in range(3): # 3 candidates
            ic = interval_categories[key_candidates[i,n],i,:]
            if ic[2] > ic_threshold and ic[3] > ic_threshold and ic[4] > ic_threshold:
               candidates.append(key_candidates[i,n])
        keys.append(candidates) 
    return keys

def createChordTemplates():
    """ creates chord templates of all triads in a key
        returns 
            templates: a 3 dimensional array for the templates
                dim: (pitch_class index,chordnumber,chromavector) -> (12x7x12)
            labels: a 2 dimensional list (12x7) containing the matching chord labels
                for the templates
    """
    notes = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
    major = mir_eval.chord.quality_to_bitmap("maj")
    minor = mir_eval.chord.quality_to_bitmap("min")
    dim = mir_eval.chord.quality_to_bitmap("dim")
    labels = []
    templates = np.zeros((12,7,12),dtype=float)
    for i in range(12):
        I = np.roll(major,i)
        II = np.roll(minor,i+2)
        III = np.roll(minor,i+4)
        IV = np.roll(major,i+5)
        V = np.roll(major,i+7)
        VI = np.roll(minor,i+9)
        VII = np.roll(dim,i+11)
        labels.append([f"{notes[i]}:maj",f"{notes[(i+2)%12]}:min",f"{notes[(i+4)%12]}:min",
                    f"{notes[(i+5)%12]}:maj",f"{notes[(i+7)%12]}:maj",f"{notes[(i+9)%12]}:min",f"{notes[(i+11)%12]}:dim"])
        chroma = np.array([I,II,III,IV,V,VI,VII],dtype=float)
        max_val = np.sum(chroma, axis=1)
        templates[i,:,:] = chroma / (np.expand_dims(max_val, axis=1)+np.finfo(float).eps)
    return (templates,labels)

def estimateChordLabels(t,chroma,pitch_class_index_key):
    """estimate the chord for each timestep in the chromagram by evaluating the angular differences
      between the projected chromavector and chord prototypes in two circles of the pitch space"""
    if chroma.ndim == 1:
        chroma = np.reshape(chroma, (1, 12))
    elif chroma.shape[1] != 12:
        raise ValueError("Array shape must be (X, 12).")
    
    # calculate angles of triad prototypes for chord estimation
    templates, labels = createChordTemplates()
    dphi_FR = np.zeros((7,))
    dphi_TR = np.zeros((7,))
    # only the template chord of the key c-major is used (pitch_class index=0)
    x_F, x_FR, x_TR, x_DR = transformChroma(templates[0,:,:])
    _, rho_FR, rho_TR, _ = transformChroma(chroma)
    est_labels = [] # estimated chord lables
    
    for time_index in range(chroma.shape[0]):
        # extract feature in the estimated key for the current timestep
        pc_index_key = pitch_class_index_key[time_index]
        r_FR = rho_FR[time_index, pc_index_key]
        r_TR = rho_TR[time_index, pc_index_key]
        # compute the angle difference for all template chords
        for template in range(7):
            # compute angle difference
            dphi_FR[template] = angle_between(r_FR,x_FR[template,0])
            dphi_TR[template] = angle_between(r_TR,x_TR[template,0])
        # pick the chordprototype with the minimum distance 
        d = np.argmin(dphi_FR+dphi_TR)
        # access the correct label for the current key and chordprototype
        est_labels.append(labels[pc_index_key][d]) 
    return est_labels

def plotHalftoneGrid(axis,n_ht):
    # axis.plot([-1,1], [0,0], color='grey',linestyle=(0,(5,10)),linewidth=1)
    # axis.plot([0,0], [-1,1], color='grey',linestyle=(0,(5,10)),linewidth=1)
    for i in np.linspace(0,n_ht,n_ht+1):
        ht = np.exp(-1j*2*np.pi*(i/n_ht))*1j
        axis.plot(ht.real,ht.imag,'o',color='grey',markersize=1.5)

def scaleVector(x,y,alpha):
    m,phi = cartesian2polar(x,y)
    m_new = m * alpha
    return polar2cartesian(m_new,phi)

def cartesian2polar(x,y,angle='rad'):
    if angle == 'deg':
        return np.sqrt(x**2+y**2),np.arctan2(y,x)*180/np.pi
    else:
        return np.sqrt(x**2+y**2),np.arctan2(y,x)

def polar2cartesian(m,phi,angle='rad'):
    if angle == 'deg':
        return m*np.cos(phi),m*np.sin(phi*np.pi/180)
    else:
        return m*np.cos(phi),m*np.sin(phi)
    
def drawLabel(axis,z,note_label,radius = 1.25,fontsize=7,tex=True):
    x,y= scaleVector(z.real,z.imag,radius)
    axis.text(x,y,note_label,rotation=np.arctan2(y,x)*180/np.pi-90,
              fontsize=fontsize,horizontalalignment='center',verticalalignment='center')

def plotCircleF(ax):
    plotHalftoneGrid(ax,84)
    for pitch_class in pitch_classes:
        n_f = sym3(49*pitch_class.chromatic_index,84,0)
        rho_F = np.exp(-1j*2*np.pi*(n_f/84))*1j
        ax.plot(rho_F.real,rho_F.imag,'o',color='k',markersize=3)
        drawLabel(ax,rho_F,pitch_class.name)
    ax.text(-1.1,1,"F",fontsize=10,horizontalalignment='center')
    ax.axis('off')

def plotCircleFR(ax,pitch_class_index=0):
    plotHalftoneGrid(ax,48)
    n_k = pitch_classes[pitch_class_index].num_accidentals
    ax.text(-1.1,1.2,f"FR({pitch_classes[pitch_class_index].name})",fontsize=8,
              horizontalalignment='center',verticalalignment='center')
    for pitch_class in pitch_classes:
        n_f = sym3(49*pitch_class.chromatic_index,84,7*n_k)
        # check if pitch is in fifth related circle
        if -21 <= (n_f-7*n_k) <= 21:
            n_fr = sym(n_f-7*n_k, 48)
            rho_FR = np.exp(-1j*2*np.pi*(n_fr/48))*1j
            ax.plot(rho_FR.real,rho_FR.imag,'ok',markersize=3)
            drawLabel(ax,rho_FR,pitch_class.name)
    ax.axis('off')

def plotCircleTR(ax,pitch_class_index=0,alterations=True):
    plotHalftoneGrid(ax,24)
    n_k = pitch_classes[pitch_class_index].num_accidentals
    if alterations:
        n_k_sharp = pitch_classes[(pitch_class_index-1)%12].num_accidentals

    ax.text(-1.1,1.2,f"TR({pitch_classes[pitch_class_index].name})",fontsize=8,
              horizontalalignment='center',verticalalignment='center')
    for pitch_class in pitch_classes:
        n_f = sym3(49*pitch_class.chromatic_index,84,7*n_k)
        # check if pitch is in fifth related circle
        if -21 <= (n_f-7*n_k) <= 21:
            n_tr = sym(n_f-7*n_k-12,24)
            rho_TR = np.exp(-1j*2*np.pi*((n_tr)/24))*1j
            ax.plot(rho_TR.real,rho_TR.imag,'ok',markersize=3)
            drawLabel(ax,rho_TR,pitch_class.name)
            continue
        if alterations:
            n_f_sharp = sym3(49*pitch_class.chromatic_index,84,7*n_k_sharp)
            n_tr_sharp = sym(n_f_sharp-7*n_k-12,24)
            r_tr_sharp = np.exp(-1j*2*np.pi*((n_tr_sharp)/24))*1j
            drawLabel(ax,r_tr_sharp,pitch_class.name)
    ax.axis('off')

def plotCircleDR(ax,pitch_class_index=0):
    plotHalftoneGrid(ax,12)
    n_k = pitch_classes[pitch_class_index].num_accidentals
    ax.text(-1.1,1.2,f"DR({pitch_classes[pitch_class_index].name})",fontsize=8,
              horizontalalignment='center',verticalalignment='center')
    for pitch_class in pitch_classes:
        n_f = sym3(49*pitch_class.chromatic_index,84,7*n_k)
        # check if pitch is in fifth related circle
        if -21 <= (n_f-7*n_k) <= 21:
            n_dr = sym(n_f-7*n_k,12)
            rho_DR = np.exp(-1j*2*np.pi*(n_dr/12))*1j
            ax.plot(rho_DR.real,rho_DR.imag,'ok',markersize=3)
            drawLabel(ax,rho_DR,pitch_class.name)          
    ax.axis('off')


def plotDiatonicTriads(ax,pitch_class_index=0,tonality="major"):
    major = mir_eval.chord.quality_to_bitmap("maj")
    minor = mir_eval.chord.quality_to_bitmap("min")
    dim = mir_eval.chord.quality_to_bitmap("dim")
    aug = mir_eval.chord.quality_to_bitmap("aug")
    if tonality == "major":
        n_k = pitch_class_index
        title = f'{pitch_classes[pitch_class_index].name} major   '
        labels = [f"{pitch_classes[pitch_class_index].name}",
                f"{pitch_classes[(pitch_class_index + 2) % 12].name}:min",
                f"{pitch_classes[(pitch_class_index + 4) % 12].name}:min",
                f"{pitch_classes[(pitch_class_index + 5) % 12].name}",
                f"{pitch_classes[(pitch_class_index + 7) % 12].name}",
                f"{pitch_classes[(pitch_class_index + 9) % 12].name}:min",
                f"{pitch_classes[(pitch_class_index + 11) % 12].name}:dim"
                ]
        chords = [np.roll(major, pitch_class_index),
                np.roll(minor, pitch_class_index + 2),
                np.roll(minor, pitch_class_index + 4),
                np.roll(major, pitch_class_index + 5),
                np.roll(major, pitch_class_index + 7),
                np.roll(minor, pitch_class_index + 9),
                np.roll(dim, pitch_class_index + 11)
                ]
        chords = np.vstack(chords)
    elif tonality=="harmonic minor":
        # a minor key is represented in its major parallel (three halftones up)
        n_k = (pitch_class_index + 3) % 12
        title = f'{pitch_classes[pitch_class_index].name} minor harm '
        labels = [f"{pitch_classes[pitch_class_index].name}m",
                f"{pitch_classes[(pitch_class_index + 2) % 12].name}°",
                f"{pitch_classes[(pitch_class_index + 3) % 12].name}+",
                f"{pitch_classes[(pitch_class_index + 5) % 12].name}m",
                f"{pitch_classes[(pitch_class_index + 7) % 12].name}m",
                f"{pitch_classes[(pitch_class_index + 8) % 12].name}",
                f"{pitch_classes[(pitch_class_index + 11) % 12].name}°"
                ]
        chords = [np.roll(minor, pitch_class_index),
                np.roll(dim, pitch_class_index + 2),
                np.roll(aug, pitch_class_index + 3),
                np.roll(minor, pitch_class_index + 5),
                np.roll(major, pitch_class_index + 7),
                np.roll(major, pitch_class_index + 8),
                np.roll(dim, pitch_class_index + 11)
                ]
        chords = np.vstack(chords)
    else:
        # a minor key is represented in its major parallel (three halftones up)
        n_k = (pitch_class_index + 3) % 12
        title = f'{pitch_classes[pitch_class_index].name} minor   '
        labels = [f"{pitch_classes[pitch_class_index].name}m",
                f"{pitch_classes[(pitch_class_index + 2) % 12].name}°",
                f"{pitch_classes[(pitch_class_index + 3) % 12].name}",
                f"{pitch_classes[(pitch_class_index + 5) % 12].name}m",
                f"{pitch_classes[(pitch_class_index + 7) % 12].name}m",
                f"{pitch_classes[(pitch_class_index + 8) % 12].name}",
                f"{pitch_classes[(pitch_class_index + 10) % 12].name}"
                ]
        chords = [np.roll(minor, pitch_class_index),
                np.roll(dim, pitch_class_index + 2),
                np.roll(major, pitch_class_index + 3),
                np.roll(minor, pitch_class_index + 5),
                np.roll(minor, pitch_class_index + 7),
                np.roll(major, pitch_class_index + 8),
                np.roll(major, pitch_class_index + 10)
                ]
        chords = np.vstack(chords)

    plotCircleF(ax[0])
    plotCircleFR(ax[1],n_k)
    plotCircleTR(ax[2],n_k)  
    r_F,r_FR,r_TR,_ = transformChroma(chords) 


    for i in range(len(chords)):
        # rotate vector for plot
        z = r_F[i] * 1j
        ax[0].plot([0, z.real], [0, z.imag], color=utilities.getColor(labels[i]), markersize=4)
        z = r_FR[i, n_k] * 1j
        ax[1].plot([0, z.real], [0, z.imag], color=utilities.getColor(labels[i]), markersize=4)
        z = r_TR[i, n_k] * 1j
        ax[2].plot([0, z.real], [0, z.imag], color=utilities.getColor(labels[i]), markersize=4)

    legend_handles = [matplotlib.patches.Patch(color=utilities.getColor(label)) for label in labels]
    legend = ax[3].legend(legend_handles, labels, loc='center left', bbox_to_anchor=(0, 0.5), title=title, handlelength=1, handletextpad=0.5, fontsize=8, title_fontsize=10, facecolor='lightgray', framealpha=0.8)
    ax[3].add_artist(legend)
    for x in ax:
        x.set_xlim(-1.5,1.5)
        x.set_ylim(-1.5,1.5)
    ax[3].axis("off")
