import numpy as np
import mir_eval
import matplotlib.pyplot as plt
from collections import namedtuple
from matplotlib.patches import Patch
import utils

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

def enharmonicNote(note):
    try:
        return enharmonic_notes[note]
    except KeyError:
        return note

def calculateDistance(x1,x2) -> float:
    e = (x1-x2)
    return np.sqrt(e.real**2+e.imag**2)

def extractNotes(chromavector,key,max_notes=12):
    # sort notes in ascending order and pick find number of n notes until a certain energy is accumulated
    notes = [(i,energy) for i,energy in zip(range(12),chromavector)]
    notes = sorted(notes, key=lambda x: x[1], reverse=True)
    for n in range(1,13):
        if sum(x[1] for x in notes[:n]) > 0.95:
            print(n)
            break
    n_k = pitch_classes[key].num_accidentals
    diatonic_notes = []
    non_harmonic_notes = []
    # split n notes into diatonic and non-harmonic notes
    for pc_index,energy in notes[:n]:
        n_c = pitch_classes[pc_index].chromatic_index
        n_f = sym3(49*n_c,84,7*n_k)
        if checkIndex(n_f,n_k):
            diatonic_notes.append(pc_index)
        else:
            non_harmonic_notes.append(pc_index)
    return diatonic_notes, non_harmonic_notes

def transformChroma(chroma,alterations=True):
    """explain away
    returns some ndarrays..
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
                if checkIndex(n_f,n_k):
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

def createChordTemplates(type="triads",root_only=False):
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
        if root_only:
            labels.append([notes[i],notes[(i+2)%12],notes[(i+4)%12],
                    notes[(i+5)%12],notes[(i+7)%12],notes[(i+9)%12],notes[(i+11)%12]])
        else:
            labels.append([f"{notes[i]}:maj",f"{notes[(i+2)%12]}:min",f"{notes[(i+4)%12]}:min",
                        f"{notes[(i+5)%12]}:maj",f"{notes[(i+7)%12]}:maj",f"{notes[(i+9)%12]}:min",f"{notes[(i+11)%12]}:dim"])
        chroma = np.array([I,II,III,IV,V,VI,VII],dtype=float)
        max_val = np.sum(chroma, axis=1)
        templates[i,:,:] = chroma / (np.expand_dims(max_val, axis=1)+np.finfo(float).eps)
    return (templates,labels)

def estimateKey(chroma,alpha=0.9,threshold=0.7,angle_weight=0.6):
    """estimate key by lowpass filtering of pitch class energy"""
    pc_energy = getPitchClassEnergyProfile(chroma,threshold=threshold,angle_weight=angle_weight)
    pc_energy_filtered = np.zeros_like(pc_energy)
    pc_energy_filtered[0,:] = pc_energy[0,:] 
    for i in range(1,pc_energy.shape[0]):
        pc_energy_filtered[i,:] = alpha * pc_energy_filtered[i-1,:] + (1-alpha) * pc_energy[i,:]
    keys = np.argsort(pc_energy_filtered,axis=1)
    #key = np.argmax(pc_energy_filtered,axis=1)
    return keys

def transcribeChromagram(t_chroma, chroma, postprocessing = False):
    def postprocessing(chroma, est_intervals, est_labels):
        entropy = -np.sum(np.multiply(chroma,np.log2(chroma+np.finfo(float).eps)), axis=1)
        intervals = []
        labels = []
        skip = []
        for i,x in enumerate(zip(est_intervals,est_labels)):
            interval,label = x
            index_0 = np.argwhere(t_chroma >= interval[0])[0][0]
            index_1 = np.argwhere(t_chroma >= interval[1])[0][0]
            try:
                mean_entropy = np.mean(entropy[index_0:index_1])
                if interval[1]-interval[0] < 0.5 or mean_entropy < 1.2:
                    skip.append(i)
                    est_intervals[i-1][1] = est_intervals[i+1][0]  # adjust interval bounds 
            except IndexError:
                continue
            
        for i in range(len(est_labels)):
            if i in skip:
                continue
            else:
                labels.append(est_labels[i])
                intervals.append(est_intervals[i])
        intervals = np.array(intervals)
        return intervals,labels
    
    max_val = np.sum(chroma, axis=1) # norm to 1
    chroma = chroma / (np.expand_dims(max_val, axis=1)+np.finfo(float).eps)
    labels = []
    templates,template_labels = createChordTemplates()
    x_F,x_FR,x_TR,x_DR = transformChroma(chroma)
    for t in range(chroma.shape[0]):
        # measure distance to each angle in the two circles
        distance = np.zeros((7,),dtype=float)
        pitch_class_index = estimateKey(chroma[t,:],0.1)
        r_F,r_FR,r_TR,r_DR = transformChroma(templates[pitch_class_index,:,:])
        for i in range(7): # for C-major templates
            d_F = calculateDistance(x_F[t],r_F[i])
            d_FR = calculateDistance(x_FR[t,pitch_class_index],r_FR[i,pitch_class_index])
            d_TR = calculateDistance(x_TR[t,pitch_class_index],r_TR[i,pitch_class_index])
            d_DR = calculateDistance(x_DR[t,pitch_class_index],r_DR[i,pitch_class_index])
            distance[i] = d_FR + d_TR
        labels.append(template_labels[pitch_class_index][np.argmin(distance)])

    # calculate chord intervals
    intervals = []
    est_labels = []
    t_start = 0
    for i,(label_t0,label_t1) in enumerate(zip(labels[:-1],labels[1:])):
        if label_t0 != label_t1: # potential chord change
                est_labels.append(label_t0)
                intervals.append([t_start,t_chroma[i]])
                t_start = t_chroma[i]
    # skip false chord changes
    est_intervals = np.array(intervals)
    if postprocessing:
        est_intervals,est_labels = postprocessing(chroma,est_intervals,est_labels)
    return est_intervals,est_labels
    
def getPitchClassEnergyProfile(chroma,threshold=0.6,angle_weight=0.5):
    """
    divide each chroma bin energy by the total chroma energy and apply thresholding of 90% by default
       angle weighting puts more emphasis on tonic of a pitch class (O .. 1). 
       This is necessary because a C-Major chord is present in pitch classes C,F and G
    """
    angle_weighting = lambda x : -((1-angle_weight)/np.pi) * np.abs(x) + 1
    pitch_class_energy = np.zeros_like(chroma)

    chroma_energy = np.square(chroma)
    total_energy = np.expand_dims(np.sum(chroma_energy,axis=1),axis=1)

    for pitch_class in pitch_classes:
        for chroma_bin in range(12):
                # iterate all chromatic indices for every pitch class and check if pitch class is present in this key
                n_c = pitch_classes[chroma_bin].chromatic_index
                n_f = sym3(49*n_c,84,7*pitch_class.num_accidentals)
                if checkIndex(n_f,pitch_class.num_accidentals):
                    n_tr = sym(n_f-7*pitch_class.num_accidentals-12,24)
                    angle = np.angle(np.exp(-1j*2*np.pi*(n_tr/24)))
                    pitch_class_energy[:,pitch_class.pitch_class_index] += angle_weighting(angle) * chroma_energy[:,chroma_bin]

    # apply tresholding for pitchclasses with low relative energy
    pitch_class_energy[pitch_class_energy < threshold * total_energy] = 0      

    pitch_class_energy = pitch_class_energy / (np.expand_dims(np.sum(pitch_class_energy,axis=1),axis=1)+np.finfo(float).eps)
    return pitch_class_energy

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
    
def fillCircle(ax,start_angle,end_angle,color='r'):
    """Color Circle for Visual Representation"""
    circle = plt.Circle([0,0], 1, fill=False)
    ax.add_artist(circle)
    theta = np.linspace(start_angle, end_angle,100)
    x = np.cos(theta)
    y = np.sin(theta)
    ax.fill_between(x, y, 0, color=color,alpha=0.4)

def sym(n,g):
    '''SYM-operator symmetriemodell'''
    try:   
        return int(np.mod((n+(g/2)),g)-(g/2))
    except ValueError:
        return None 
    
def sym3(n,g,n0):
    x = sym(n-n0,g)
    if x is not None:
        return x+n0
    else:
        return None

def drawLabel(axis,z,note_label,radius = 1.25,fontsize=7,tex=True):
    x,y= scaleVector(z.real,z.imag,radius)
    axis.text(x,y,utils.formatChordLabel(note_label),rotation=np.arctan2(y,x)*180/np.pi-90,
              fontsize=fontsize,horizontalalignment='center',verticalalignment='center')


def plotHalftoneGrid(axis,n_ht):
    # axis.plot([-1,1], [0,0], color='grey',linestyle=(0,(5,10)),linewidth=1)
    # axis.plot([0,0], [-1,1], color='grey',linestyle=(0,(5,10)),linewidth=1)
    x = np.linspace(0,n_ht,n_ht+1)
    for i in x:
        ht = np.exp(-1j*2*np.pi*(i/n_ht))*1j
        axis.plot(ht.real,ht.imag,'o',color='grey',markersize=1.5)

def plotVector(axis,z,norm=False,**kwargs):
    # rotate for CPS plots
    z = z*1j
    if norm:
        z = np.exp(1j*np.angle(z))
    axis.arrow(0,0,z.real,z.imag,**kwargs)

def calculateVectorsFR(pitch_class_index=0):
    n_k = pitch_classes[pitch_class_index].num_accidentals
    vectors_FR = []
    for pitch_class in pitch_classes:
        n_f = sym3(49*pitch_class.chromatic_index,84,7*n_k)
        # check if pitch is in fifth related circle
        if checkIndex(n_f,n_k):
            n_fr = sym(n_f-7*n_k, 48)
            vectors_FR.append(np.exp(-1j*2*np.pi*(n_fr/48)))
    return vectors_FR

def calculateVectorsTR(pitch_class_index=0):
    n_k = pitch_classes[pitch_class_index].num_accidentals
    for pitch_class in pitch_classes:
        n_f = sym3(49*pitch_class.chromatic_index,84,7*n_k)
        # check if pitch is in fifth related circle
        if checkIndex(n_f,n_k):
            n_tr = sym(n_f-7*n_k-12,24)
            rho_TR = np.exp(-1j*2*np.pi*((n_tr)/24))

def plotCircleOfFifths(ax):
    plotHalftoneGrid(ax,84)
    for pitch_class in pitch_classes:
        n_f = sym3(49*pitch_class.chromatic_index,84,0)
        rho_F = np.exp(-1j*2*np.pi*(n_f/84))*1j
        ax.plot(rho_F.real,rho_F.imag,'o',color='k',markersize=3)
        drawLabel(ax,rho_F,pitch_class.name)
    ax.text(-1.1,1,"F",fontsize=10,horizontalalignment='center')
    ax.axis('off')

def plotCircleOfFifthsRelated(ax,pitch_class_index=0):
    plotHalftoneGrid(ax,48)
    n_k = pitch_classes[pitch_class_index].num_accidentals
    ax.text(-1.1,1.2,f"FR({pitch_classes[pitch_class_index].name})",fontsize=8,
              horizontalalignment='center',verticalalignment='center')
    for pitch_class in pitch_classes:
        n_f = sym3(49*pitch_class.chromatic_index,84,7*n_k)
        # check if pitch is in fifth related circle
        if checkIndex(n_f,n_k):
            n_fr = sym(n_f-7*n_k, 48)
            rho_FR = np.exp(-1j*2*np.pi*(n_fr/48))*1j
            ax.plot(rho_FR.real,rho_FR.imag,'ok',markersize=3)
            drawLabel(ax,rho_FR,pitch_class.name)
    ax.axis('off')

def plotCircleOfThirdsRelated(ax,pitch_class_index=0,alterations=True):
    plotHalftoneGrid(ax,24)
    n_k = pitch_classes[pitch_class_index].num_accidentals
    if alterations:
        n_k_sharp = pitch_classes[(pitch_class_index-1)%12].num_accidentals

    ax.text(-1.1,1.2,f"TR({pitch_classes[pitch_class_index].name})",fontsize=8,
              horizontalalignment='center',verticalalignment='center')
    for pitch_class in pitch_classes:
        n_f = sym3(49*pitch_class.chromatic_index,84,7*n_k)
        # check if pitch is in fifth related circle
        if checkIndex(n_f,n_k):
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

def plotCircleOfDiatonicRelated(ax,pitch_class_index=0):
    plotHalftoneGrid(ax,12)
    n_k = pitch_classes[pitch_class_index].num_accidentals
    ax.text(-1.1,1.2,f"DR({pitch_classes[pitch_class_index].name})",fontsize=8,
              horizontalalignment='center',verticalalignment='center')
    for pitch_class in pitch_classes:
        n_f = sym3(49*pitch_class.chromatic_index,84,7*n_k)
        # check if pitch is in fifth related circle
        if checkIndex(n_f,n_k):
            n_dr = sym(n_f-7*n_k,12)
            rho_DR = np.exp(-1j*2*np.pi*(n_dr/12))*1j
            ax.plot(rho_DR.real,rho_DR.imag,'ok',markersize=3)
            drawLabel(ax,rho_DR,pitch_class.name)          
    ax.axis('off')

def checkIndex(n_f,n_k): 
    """Checks if realtone index n_f is present in the key with n_k accidentals"""
    if n_f is not None and -21 <= (n_f-7*n_k) and (n_f-7*n_k) <= 21:
        return True
    else:
        return False
    
def plotPitchSpace():
    fig_cps = plt.figure(figsize=((8.27,11.69)))
    grid = plt.GridSpec(7, 7, wspace=0.55, hspace=0.55)
    axes_list = [[] for _ in range(7)]
    for i in range(7):
        ax = fig_cps.add_subplot(grid[0,i])
        ax.axis('off')
        if i == 3:
            plotCircleOfFifths(ax)
        axes_list[0].append(ax)
    # plot first 7 features in 1,3,5 row of the grid
    for i in range(7):
        ax = fig_cps.add_subplot(grid[1,i])
        plotCircleOfFifthsRelated(ax,i)
        axes_list[1].append(ax)
        ax = fig_cps.add_subplot(grid[3,i])
        plotCircleOfThirdsRelated(ax,i)
        axes_list[3].append(ax)
        ax = fig_cps.add_subplot(grid[5,i])
        plotCircleOfDiatonicRelated(ax,i)
        axes_list[5].append(ax)

    # add empty axes to the start of row 2,4,6
    for x in (2,4,6):
        axes_list[x].append(None)
    for i in range(7,12):
        ax = fig_cps.add_subplot(grid[2,i-6])
        plotCircleOfFifthsRelated(ax,i)
        axes_list[2].append(ax)
        ax = fig_cps.add_subplot(grid[4,i-6])
        plotCircleOfThirdsRelated(ax,i)
        axes_list[4].append(ax)
        ax = fig_cps.add_subplot(grid[6,i-6])
        plotCircleOfDiatonicRelated(ax,i)
        axes_list[6].append(ax)
    # append empty axes at the end of row 2,4,6
    for x in (2,4,6):
        axes_list[x].append(None)
    return fig_cps,axes_list

def plotFeatures(ax_list,rho_F,rho_FR,rho_TR,rho_DR,color='r'):
    kwargs = {'head_width':0.1,'head_length':0.1,'color':color}
    plotVector(ax_list[0][3],rho_F,**kwargs)
    for i in range(12):
        if i < 7:
            plotVector(ax_list[1][i],rho_FR[i],**kwargs)
            plotVector(ax_list[3][i],rho_TR[i],**kwargs)    
            plotVector(ax_list[5][i],rho_DR[i],**kwargs)
        else:
            plotVector(ax_list[2][i-6],rho_FR[i],**kwargs)
            plotVector(ax_list[4][i-6],rho_TR[i],**kwargs)    
            plotVector(ax_list[6][i-6],rho_DR[i],**kwargs)

def plotDiatonicTriads(ax,pitch_class_index=0,tonality="major"):
    major = mir_eval.chord.quality_to_bitmap("maj")
    minor = mir_eval.chord.quality_to_bitmap("min")
    dim = mir_eval.chord.quality_to_bitmap("dim")
    aug = mir_eval.chord.quality_to_bitmap("aug")
    colors = ["blue", "green", "red", "orange", "purple", "grey", "black"]
    if tonality == "major":
        n_k = pitch_class_index
        title = f'{pitch_classes[pitch_class_index].name} major   '
        labels = [f"{pitch_classes[pitch_class_index].name}",
                f"{pitch_classes[(pitch_class_index + 2) % 12].name}m",
                f"{pitch_classes[(pitch_class_index + 4) % 12].name}m",
                f"{pitch_classes[(pitch_class_index + 5) % 12].name}",
                f"{pitch_classes[(pitch_class_index + 7) % 12].name}",
                f"{pitch_classes[(pitch_class_index + 9) % 12].name}m",
                f"{pitch_classes[(pitch_class_index + 11) % 12].name}°"
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

    plotCircleOfFifthsRelated(ax[0],n_k)
    plotCircleOfThirdsRelated(ax[1],n_k)
    plotCircleOfDiatonicRelated(ax[2],n_k)  

    _,r_FR,r_TR,r_DR = transformChroma(chords) 


    for i in range(len(chords)):
        # rotate vector for plot
        z = r_FR[i, n_k] * 1j
        ax[0].plot([0, z.real], [0, z.imag], color=colors[i], markersize=4)
        z = r_TR[i, n_k] * 1j
        ax[1].plot([0, z.real], [0, z.imag], color=colors[i], markersize=4)
        z = r_DR[i, n_k] * 1j
        ax[2].plot([0, z.real], [0, z.imag], color=colors[i], markersize=4)

    legend_handles = [Patch(color=color) for color in colors]
    legend = ax[3].legend(legend_handles, labels, loc='center left', bbox_to_anchor=(0, 0.5), title=title, handlelength=1, handletextpad=0.5, fontsize=8, title_fontsize=10, facecolor='lightgray', framealpha=0.8)
    ax[3].add_artist(legend)
    for x in ax:
        x.set_xlim(-1.5,1.5)
        x.set_ylim(-1.5,1.5)
    ax[3].axis("off")

def plotDiatonicTetrads(pitch_class_index=0,tonality="major"):
    fig, ax = plt.subplots(1, 4, figsize=(11, 11/3))
    maj7 = mir_eval.chord.quality_to_bitmap("maj7")
    min7 = mir_eval.chord.quality_to_bitmap("min7")
    hdim7 = mir_eval.chord.quality_to_bitmap("hdim7")
    dim7 = mir_eval.chord.quality_to_bitmap("dim7")

    colors = ["blue", "green", "red", "orange", "purple", "grey", "black"]
    if tonality == "major":
        n_k = pitch_class_index
        title = f'Triads of {pitch_classes[pitch_class_index].name} major'
        labels = [f"{pitch_classes[pitch_class_index].name}maj7",
                f"{pitch_classes[(pitch_class_index + 2) % 12].name}min7",
                f"{pitch_classes[(pitch_class_index + 4) % 12].name}min7",
                f"{pitch_classes[(pitch_class_index + 5) % 12].name}maj7",
                f"{pitch_classes[(pitch_class_index + 7) % 12].name}7",
                f"{pitch_classes[(pitch_class_index + 9) % 12].name}min7",
                f"{pitch_classes[(pitch_class_index + 11) % 12].name}hdim7"
                ]
        chords = [np.roll(maj7, pitch_class_index),
                np.roll(min7, pitch_class_index + 2),
                np.roll(min7, pitch_class_index + 4),
                np.roll(maj7, pitch_class_index + 5),
                np.roll(maj7, pitch_class_index + 7),
                np.roll(min7, pitch_class_index + 9),
                np.roll(hdim7, pitch_class_index + 11)
                ]
        chords = np.vstack(chords)
    plotCircleOfFifths(ax[0])
    plotCircleOfFifthsRelated(ax[1],n_k)
    plotCircleOfThirdsRelated(ax[2],n_k)
    plotCircleOfDiatonicRelated(ax[3],n_k)  

    r_F,r_FR,r_TR,r_DR = transformChroma(chords,enharmonic=True) 
    for i in range(len(chords)):
        # rotate vector for plot
        z = r_F[i] * 1j
        ax[0].plot([0, z.real], [0, z.imag], color=colors[i], markersize=2)
        z = r_FR[i, n_k] * 1j
        ax[1].plot([0, z.real], [0, z.imag], color=colors[i], markersize=2)
        z = r_TR[i, n_k] * 1j
        ax[2].plot([0, z.real], [0, z.imag], color=colors[i], markersize=2)
        z = r_DR[i, n_k] * 1j
        ax[3].plot([0, z.real], [0, z.imag], color=colors[i], markersize=2)

    legend_handles = [Patch(color=color) for color in colors]
    fig.subplots_adjust(top=0.8) 
    fig.legend(legend_handles, labels, loc='upper center', ncol=len(labels),
                title=title, handlelength=1, handletextpad=0.5, fontsize=8,
                  title_fontsize=10, facecolor='lightgray', framealpha=0.8)

    for x in ax:
        x.set_xlim(-1.5,1.5)
        x.set_ylim(-1.5,1.5)

    plt.show()


if __name__=="__main__":
    fig,ax = plt.subplots(3,4,figsize=(11,9))
    plotCircleOfThirdsRelated(ax[0,0],1)
    plotCircleOfThirdsRelated(ax[0,1],0)
    plotCircleOfThirdsRelated(ax[0,2],11)
    plt.show()