import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple

PitchClass = namedtuple('PitchClass','name pitch_class_index chromatic_index num_accidentals')
""" 
    Pitch class
    pitch_class_index : index of pitch class in chroma vector and list of pitch_classes
    chromatic_index : index n_c for pitch class in pitch space 
    num_accidentals : The number of accidentals n_k present in the key of this pitch class 
    pitchspace_index : index of pitch class in array of vectors in r_FR,r_TR and r_DR 
"""

pitch_classes = [
            PitchClass("C",0,-2,0),
            PitchClass("C#",1,-1,-5), # use enharmonic note with lowest accidentals (Db)! (C# has 7 crosses) 
            PitchClass('D',2,0,2),
            PitchClass("D#",3,1,-3),  #Eb
            PitchClass("E",4,2,4),
            PitchClass("F",5,3,-1),
            PitchClass("F#",6,4,6),
            PitchClass("G",7,5,1),
            PitchClass("G#",8,-6,-4), # Ab
            PitchClass("A",9,-5,3),
            PitchClass("A#",10,-4,-2), #Bb
            PitchClass("B",11,-3,5)
]

def getPitchClassEnergyProfile(chroma,threshold=0.6,weighting=0.7):
    """divide each chroma bin energy by the total chroma energy and apply thresholding of 90% by default
       angle weighting puts more emphasis on tonic of a pitch class. e.g a C-Major chord is present pitch classes C,F and G
    """
    angle_weighting = lambda x : -((1-weighting)/np.pi) * np.abs(x) + 1
    pitch_class_energy = np.zeros_like(chroma)

    chroma_energy = np.square(chroma)
    total_energy = np.reshape(np.repeat(np.sum(chroma_energy,axis=1),12),chroma.shape)
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
    pitch_class_energy = np.multiply(pitch_class_energy,1/total_energy)
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

def drawLabel(axis,z,note_label,radius = 1.2,fontsize=7):
    x,y= scaleVector(z.real,z.imag,radius)
    axis.text(x,y,note_label,rotation=np.arctan2(y,x)*180/np.pi-90,
              fontsize=fontsize,horizontalalignment='center',verticalalignment='center')

def plotHalftoneGrid(axis,n_ht):
    axis.plot([-1,1], [0,0], color='grey',linestyle=(0,(5,10)),linewidth=1)
    axis.plot([0,0], [-1,1], color='grey',linestyle=(0,(5,10)),linewidth=1)
    x = np.linspace(0,n_ht,n_ht+1)
    for i in x:
        ht = np.exp(-1j*2*np.pi*(i/n_ht))*1j
        axis.plot(ht.real,ht.imag,'o',color='grey',markersize=1.5)

def plotVector(axis,vec,rotation=np.pi/2,**kwargs):
    # rotate for CPS plots
    x,y = polar2cartesian(vec[0],vec[1]+rotation)
    axis.arrow(0,0,x,y,**kwargs)
    axis.set_xlim((-1,1))
    axis.set_ylim((-1,1))
    axis.set_axis_off()

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

def plotCircleOfThirdsRelated(ax,pitch_class_index=0):
    plotHalftoneGrid(ax,24)
    n_k = pitch_classes[pitch_class_index].num_accidentals
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
    
def transformChroma(chroma):
    """explain away
    returns some ndarrays..
    """    
    rho_F  = np.zeros((chroma.shape[0],),dtype=complex)
    rho_FR = np.zeros_like(chroma,dtype=complex)
    rho_TR = np.zeros_like(chroma,dtype=complex)
    rho_DR = np.zeros_like(chroma,dtype=complex)
    # iterate over all time steps
    for time_index in range(chroma.shape[0]):
        # calculte key related circles
        for x in pitch_classes:
            n_k = x.num_accidentals
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
    return (rho_F,rho_FR,rho_TR,rho_DR)

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

if __name__=='__main__':
    plotPitchSpace()
    plt.show()



    
