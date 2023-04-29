import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple

PitchClass = namedtuple('PitchClass','name chroma_index chromatic_index num_accidentals pitchspace_index')
""" 
    Pitch class
    Chroma_index : index of pitch class in chroma vector
    chromatic_index : index n_c for pitch class in pitch space 
    num_accidentals : The number of accidentals present in the key of this pitch class 
    pitchspace_index : index of pitch class in array of vectors in r_FR,r_TR and r_DR 
"""

pitch_classes = [
            PitchClass("C",0,-2,0,4),
            PitchClass("C#",1,-1,-5,5), # use enharmonic note with lowest accidentals (Db)! (C# has 7 crosses) 
            PitchClass('D',2,0,2,6),
            PitchClass("D#",3,1,-3,7),  #Eb
            PitchClass("E",4,2,4,8),
            PitchClass("F",5,3,-1,9),
            PitchClass("F#",6,4,6,10),
            PitchClass("G",7,5,1,11),
            PitchClass("G#",8,-6,-4,0), # Ab
            PitchClass("A",9,-5,3,1),
            PitchClass("A#",10,-4,-2,2), #Bb
            PitchClass("B",11,-3,5,3)
]

# chromatic index of a pitch
# denoted as n_i
chromatic_index = {-6:'gis',-5:'a',-4:'b',-3:'h',-2:'c',-1:'cis',0:'d',1:'es',2:'e',3:'f',4:'fis',5:'g'}

# number of accidentals: flats (+) or sharps (-) in a key  (e.g. F-maj=-1,C-maj=0, G-Maj=1)
# denoted as n_k
num_accidentals = {-5:'Des',-4:'As',-3:'Es',-2:'B',-1:'F',0:'C',1:'G',2:'D',3:'A',4:'E',5:'H',6:'Fis'}

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
                    pitch_class_energy[:,pitch_class.chroma_index] += angle_weighting(angle) * chroma_energy[:,chroma_bin]

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

def getCircleLabel(index):
    """explain away"""
    n_k = list(num_accidentals.keys())[index]
    return num_accidentals[n_k]

def getChromaticIndices():
    """this function maps the chroma indices to the chromatic indices of the pitch space 
       for transformation of the chroma vector.
       The Chroma vector starts with the note 'C'!
       [0,1, .. ,11] -> [-2,-1 ..,-3]"""
    return np.roll(list(chromatic_index.keys()),-4)

def annotateLabel(axis,z,note_label,ht_steps=None,fontsize=7):
    x,y= scaleVector(z.real,z.imag,1.25)
    axis.text(x,y,note_label,rotation=np.arctan2(y,x)*180/np.pi-90,fontsize=fontsize,horizontalalignment='center',verticalalignment='center')
    if ht_steps:
        x,y = scaleVector(z.real,z.imag,0.8)
        axis.text(x,y,ht_steps,rotation=np.arctan2(y,x)*180/np.pi-90,fontsize=fontsize,horizontalalignment='center',verticalalignment='center')

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

def plotCircleOfFifths(axis):
    plotHalftoneGrid(axis,84)
    for n_c in chromatic_index:
        n_f = sym3(49*n_c,84,0)
        rho_F = np.exp(-1j*2*np.pi*(n_f/84))*1j
        axis.plot(rho_F.real,rho_F.imag,'o',color='k',markersize=3)
        annotateLabel(axis,rho_F,chromatic_index[n_c],None,fontsize=7)
    axis.text(-1.1,1,"F",fontsize=10,horizontalalignment='center')
    axis.axis('off')

def checkIndex(n_f,n_k):
    if n_f is not None and -21 <= (n_f-7*n_k) and (n_f-7*n_k) <= 21:
        return True
    else:
        return False
    
def plotKeyRelatedRealPitches(axis,n_k=0,circle='FR',plot_halftones=False):
    if circle == 'FR':
    # plot key related circle of fiths
        plotHalftoneGrid(axis,48)
        axis.text(-1.1,1.2,f"FR({num_accidentals[n_k]})",fontsize=8,horizontalalignment='center',verticalalignment='center')
        for n_c in chromatic_index.keys():
            n_f = sym3(49*n_c,84,7*n_k)
            # check if pitch is in fifth related circle
            if checkIndex(n_f,n_k):
                n_fr = sym(n_f-7*n_k, 48)
                rho_FR = np.exp(-1j*2*np.pi*(n_fr/48))*1j
                axis.plot(rho_FR.real,rho_FR.imag,'ok',markersize=3)
                if plot_halftones:
                    annotateLabel(axis,rho_FR,chromatic_index[n_c],n_fr)
                else:
                    annotateLabel(axis,rho_FR,chromatic_index[n_c],None)
        # plot key related circle of thirds
    elif circle == 'TR':
        plotHalftoneGrid(axis,24)
        axis.text(-1.1,1.2,f"TR({num_accidentals[n_k]})",fontsize=8,horizontalalignment='center',verticalalignment='center')
        for n_c in chromatic_index:
            n_f = sym3(49*n_c,84,7*n_k)
            # check if pitch is in fifth related circle
            if checkIndex(n_f,n_k):
                n_tr = sym(n_f-7*n_k-12,24)
                rho_TR=np.exp(-1j*2*np.pi*((n_tr)/24))*1j
                axis.plot(rho_TR.real,rho_TR.imag,'ok',markersize=3)
                if plot_halftones:
                    annotateLabel(axis,rho_TR,chromatic_index[n_c],n_tr)
                else:
                    annotateLabel(axis,rho_TR,chromatic_index[n_c],None)
    elif circle =='DR':
        # plot key related diatonic circle
        plotHalftoneGrid(axis,12)
        axis.text(-1.1,1.2,f"DR({num_accidentals[n_k]})",fontsize=8,horizontalalignment='center',verticalalignment='center')
        for n_c in chromatic_index:
            # map pitch index to circle of fifth
            n_f = sym3(49*n_c,84,7*n_k)
            # check if pitch is in fifth related circle
            if checkIndex(n_f,n_k):
                n_dr = sym(n_f-7*n_k,12)
                rho_DR=np.exp(-1j*2*np.pi*(n_dr/12))*1j
                axis.plot(rho_DR.real,rho_DR.imag,'ok',markersize=3)
                if plot_halftones:
                    annotateLabel(axis,rho_DR,chromatic_index[n_c],n_dr)
                else:
                    annotateLabel(axis,rho_DR,chromatic_index[n_c],None)
    axis.axis('off')
        
def plotChromaVector(axis,chroma,n_k,circle='F'):
    axis.set_xlim([-1,1])
    axis.set_ylim([-1,1])
    chroma_idx = getChromaticIndices()
    if circle == 'F':
        rho_F = np.zeros(1,dtype=complex)
        for i,chroma_bin in enumerate(chroma):
            n_f = sym3(49*chroma_idx[i],84,0)
            # calculate vector entries for circle of fifth
            temp = chroma_bin * np.exp(-1j*2*np.pi*(n_f/84))*1j
            axis.plot(temp.real,temp.imag,'xk')
            rho_F += temp
        axis.quiver(0,0,rho_F.real,rho_F.imag, units='xy',scale=1,color='r')
    elif circle =='FR':
        rho_FR = np.zeros(1,dtype=complex)
        for i,chroma_bin in enumerate(chroma):
            n_f = sym3(49*chroma_idx[i],84,7*n_k)
            if checkIndex(n_f,n_k):
                n_fr = sym(n_f-7*n_k, 48)
                temp = chroma_bin*np.exp(-1j*2*np.pi*(n_fr/48))*1j
                axis.plot(temp.real,temp.imag,'xk')
                rho_FR +=temp 
        axis.quiver(0,0,rho_FR.real,rho_FR.imag, units='xy',scale=1,color='r')
    elif circle =='TR':
        rho_TR = np.zeros(1,dtype=complex)
        for i,chroma_bin in enumerate(chroma):
            n_f = sym3(49*chroma_idx[i],84,7*n_k)
            if checkIndex(n_f,n_k):
                n_tr = sym(n_f-7*n_k-12,24)
                temp = chroma_bin*np.exp(-1j*2*np.pi*((n_tr)/24))*1j
                axis.plot(temp.real,temp.imag,'xk')
                rho_TR +=temp
        axis.quiver(0,0,rho_TR.real,rho_TR.imag, units='xy',scale=1,color='r')
    elif circle == 'DR':
        rho_DR = np.zeros(1,dtype=complex)
        for i,chroma_bin in enumerate(chroma):
            n_f = sym3(49*chroma_idx[i],84,7*n_k)
            if checkIndex(n_f,n_k):
                n_dr = sym(n_f-7*n_k,12)
                temp = chroma_bin*np.exp(-1j*2*np.pi*(n_dr/12))*1j
                axis.plot(temp.real,temp.imag,'xk')
                rho_DR += temp
        axis.quiver(0,0,rho_DR.real,rho_DR.imag, units='xy',scale=1,color='r')
    axis.axis('off')

def transformChroma(chroma):
    """explain away
    returns some ndarrays..
    """    
    rho_F  = np.zeros((chroma.shape[0],),dtype=complex)
    rho_FR = np.zeros_like(chroma,dtype=complex)
    rho_TR = np.zeros_like(chroma,dtype=complex)
    rho_DR = np.zeros_like(chroma,dtype=complex)
    chroma_index = getChromaticIndices()
    # iterate over all time steps
    for time_index in range(chroma.shape[0]):
        # calculte key related circles
        for key_index,n_k in enumerate(num_accidentals):
            # iterate over all chroma bins with correct index
            for chroma_i,chroma_bin in zip(chroma_index,chroma[time_index,:]):
                n_f = sym3(49*chroma_i,84,7*n_k)
                if n_k == 0:
                    # calculate vector entries for circle of fifth
                    rho_F[time_index] += chroma_bin * np.exp(-1j*2*np.pi*(n_f/84))
                # check if real pitch is part of the key n_k
                if checkIndex(n_f,n_k):
                    n_fr = sym(n_f-7*n_k, 48)
                    # fifth related circle
                    rho_FR[time_index,key_index] += chroma_bin*np.exp(-1j*2*np.pi*(n_fr/48))       
                    n_tr = sym(n_f-7*n_k-12,24)
                    # third related circle
                    rho_TR[time_index,key_index] += chroma_bin*np.exp(-1j*2*np.pi*(n_tr/24))        
                    # diatonic circle   
                    n_dr = sym(n_f-7*n_k,12)
                    rho_DR[time_index,key_index] += chroma_bin*np.exp(-1j*2*np.pi*(n_dr/12))

    #   this is useless!                
    # cartesian 2 polar conversion
    # rho_F = list(map(lambda x: cartesian2polar(x.real,x.imag), rho_F))
    # rho_FR = list(map(lambda fr: [cartesian2polar(x.real,x.imag) for x in fr], rho_FR))
    # rho_TR = list(map(lambda tr: [cartesian2polar(x.real,x.imag) for x in tr], rho_TR))
    # rho_DR = list(map(lambda dr: [cartesian2polar(x.real,x.imag) for x in dr], rho_DR))

    return (rho_F,rho_FR,rho_TR,rho_DR)

def plotPitchSpace(size="A4"):
    if size == 'A4':
        fig_cps = plt.figure(figsize=((8.27,11.69)))
    else:
        fig_cps = plt.figure(figsize=(size))
    # helper variable
    n_k = list(num_accidentals.keys())
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
        plotKeyRelatedRealPitches(ax,n_k[i],'FR')
        axes_list[1].append(ax)
        ax = fig_cps.add_subplot(grid[3,i])
        plotKeyRelatedRealPitches(ax,n_k[i],'TR')
        axes_list[3].append(ax)
        ax = fig_cps.add_subplot(grid[5,i])
        plotKeyRelatedRealPitches(ax,n_k[i],'DR')
        axes_list[5].append(ax)

    # add empty axes to the start of row 2,4,6
    for x in (2,4,6):
        axes_list[x].append(None)
    for i in range(7,12):
        ax = fig_cps.add_subplot(grid[2,i-6])
        plotKeyRelatedRealPitches(ax,n_k[i],'FR')
        axes_list[2].append(ax)
        ax = fig_cps.add_subplot(grid[4,i-6])
        plotKeyRelatedRealPitches(ax,n_k[i],'TR')
        axes_list[4].append(ax)
        ax = fig_cps.add_subplot(grid[6,i-6])
        plotKeyRelatedRealPitches(ax,n_k[i],'DR')
        axes_list[6].append(ax)
    # append empty axes at the end of row 2,4,6
    for x in (2,4,6):
        axes_list[x].append(None)
    return fig_cps,axes_list

def plotFeatures(ax_list,rho_F,rho_FR,rho_TR,rho_DR,color='r'):
    kwargs = {'head_width':0.1,'head_length':0.1,'color':color}
    n_k = list(num_accidentals.keys())
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
    chroma = np.array([[1,0,0,0,1,0,0,1,0,0,0,0],[1,0,0,0,1,0,0,0,0,1,0,0]],dtype=float)
    x= getPitchClassEnergyProfile(chroma)
    print(x)




    
