import numpy as np
import matplotlib.pyplot as plt

# chromatic index of a pitch
# denoted as n_i
chromatic_index = {-6:'gis',-5:'a',-4:'ais',-3:'h',-2:'c',-1:'cis',0:'d',1:'dis',2:'e',3:'f',4:'fis',5:'g'}

# number of accidentals: flats (+) or sharps (-) in a key  (e.g. F-maj=-1,C-maj=0, G-Maj=1)
# denoted as n_k
num_accidentals = {-5:'Des',-4:'As',-3:'Es',-2:'B',-1:'F',0:'C',1:'G',2:'D',3:'A',4:'E',5:'H',6:'Fis/Ges'}


def scaleVector(x,y,alpha):
    m,phi = cartesian2polar(x,y)
    m_new = m * alpha
    return polar2cartesian(m_new,phi)

def cartesian2polar(x,y):
    return np.sqrt(x**2+y**2),np.arctan2(y,x)

def polar2cartesian(m,phi):
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
    
def getChromaticIndices():
    """this function maps the chroma indices to the chromatic indices of the pitch space 
       for transformation of the chroma vector.
       The Chroma vector starts with the note 'C'!
       [0,1, .. ,11] -> [-2,-1 ..,-3]
    """
    n_c = np.roll(list(chromatic_index.keys()),-4)
    return n_c

def annotateLabel(axis,xy,note_label,ht_steps):
    re,im = scaleVector(xy.real,xy.imag,1.1)
    axis.text(re,im,note_label,rotation=np.arctan2(im,re)*180/np.pi-90,horizontalalignment='center',verticalalignment='center')
    re,im = scaleVector(xy.real,xy.imag,0.85)
    axis.text(re,im,ht_steps,rotation=np.arctan2(im,re)*180/np.pi-90,horizontalalignment='center',verticalalignment='center')

def plotHalftoneGrid(axis,n_ht):
    axis.plot([-1,1], [0,0], color='grey',linestyle=(0,(5,10)),linewidth=1)
    axis.plot([0,0], [-1,1], color='grey',linestyle=(0,(5,10)),linewidth=1)
    x = np.linspace(0,n_ht,n_ht+1)
    for i in x:
        ht = np.exp(-1j*2*np.pi*(i/n_ht))*1j
        axis.plot(ht.real,ht.imag,'o',color='grey',markersize=2)

def plotCircleOfFifths(ax,plot_halftones=False):
    plotHalftoneGrid(ax,84)
    for n_k in num_accidentals:
        # realtonigkeit plotten
        rho_F = np.exp(-1j*2*np.pi*(7*n_k/84))*1j
        ax.plot(rho_F.real,rho_F.imag,'o',color='k',markersize=4)
        if plot_halftones:
            annotateLabel(ax,rho_F,num_accidentals[n_k],7*n_k)
        else:
            annotateLabel(ax,rho_F,num_accidentals[n_k],None)
    ax.set_title("Circle of Fifths")

def checkIndex(n_f,n_k):
    if n_f is not None and -21 <= (n_f-7*n_k) and (n_f-7*n_k) <= 21:
        return True
    else:
        return False
    
def plotKeyRelatedRealPitches(axis,n_k,type='fifths',plot_halftones=False):
    if type=='fifths':
        plotHalftoneGrid(axis,48)
        axis.text(0,1.5,f"FR({num_accidentals[n_k]})",horizontalalignment='center',verticalalignment='center')
        for n_c in chromatic_index.keys():
            n_f = sym3(49*n_c,84,7*n_k)
            # test ob tonigkeit im tonartbezogengen Kreis ist
            if checkIndex(n_f,n_k):
                n_fr = sym(n_f-7*n_k, 48)
                rho_FR = np.exp(-1j*2*np.pi*(n_fr/48))*1j
                axis.plot(rho_FR.real,rho_FR.imag,'ok',markersize=4)
                if plot_halftones:
                    annotateLabel(axis,rho_FR,chromatic_index[n_c],n_fr)
                else:
                    annotateLabel(axis,rho_FR,chromatic_index[n_c],None)

    elif type=='thirds':
        plotHalftoneGrid(axis,24)
        axis.text(0,1.5,f"TR({num_accidentals[n_k]})",horizontalalignment='center',verticalalignment='center')
        for n_c in chromatic_index:
            n_f = sym3(49*n_c,84,7*n_k)
            # check if pitch is in fifth related circle
            if checkIndex(n_f,n_k):
                n_tr = sym(n_f-7*n_k-12,24)
                rho_TR=np.exp(-1j*2*np.pi*((n_tr)/24))*1j
                axis.plot(rho_TR.real,rho_TR.imag,'ok',markersize=4)
                if plot_halftones:
                    annotateLabel(axis,rho_TR,chromatic_index[n_c],n_tr)
                else:
                    annotateLabel(axis,rho_TR,chromatic_index[n_c],None)
    elif type=='diatonic':
        plotHalftoneGrid(axis,12)
        axis.text(0,1.5,f"DR({num_accidentals[n_k]})",horizontalalignment='center',verticalalignment='center')
        for n_c in chromatic_index:
            # map pitch index to circle of fifth
            n_f = sym3(49*n_c,84,7*n_k)
            # check if pitch is in fifth related circle
            if checkIndex(n_f,n_k):
                n_dr = sym(n_f-7*n_k,12)
                rho_DR=np.exp(-1j*2*np.pi*(n_dr/12))*1j
                axis.plot(rho_DR.real,rho_DR.imag,'ok',markersize=4)
                if plot_halftones:
                    annotateLabel(axis,rho_DR,chromatic_index[n_c],n_dr)
                else:
                    annotateLabel(axis,rho_DR,chromatic_index[n_c],None)

def plotChromaVector(axs,chroma,key_index,arrow_color='r'):
    # plot F,FR,TR,DR ->
    for ax in axs:
        ax.set_xlim([-1,1])
        ax.set_ylim([-1,1])
    # circular shift of chroma vector, adjust index
    rho_FR = np.zeros(1,dtype=complex)
    rho_TR = np.zeros(1,dtype=complex)
    rho_DR = np.zeros(1,dtype=complex)
    # halftones are spaced around -6,5 -> create index
    lst = np.linspace(0,11,12,dtype=int)
    chroma_idx = []
    for i,x in enumerate(np.roll(lst,0)):
        if(i<=7):
            chroma_idx.append(i-2)
        else:
            chroma_idx.append(i-14)
    for i,x in enumerate(chroma):
        n_f = sym3(49*chroma_idx[i],84,7*key_index)
        if checkIndex(n_f,key_index):
            # plot tonal components in circle of fifths
            # plot is rotated by 90° according to definitions (multiply by 1j)
            n_fr = sym(n_f-7*key_index, 48)
            temp = x*np.exp(-1j*2*np.pi*(n_fr/48))*1j
            axs[0].plot(temp.real,temp.imag,'x',markersize=4,color=arrow_color)
            rho_FR +=temp
            # plot tonal components in circle of thirds
            n_tr = sym(n_f-7*key_index-12,24)
            temp=x*np.exp(-1j*2*np.pi*((n_tr)/24))*1j
            axs[1].plot(temp.real,temp.imag,'x',markersize=4,color=arrow_color)
            rho_TR +=temp
            # plot tonal components in diatonic circle
            n_dr = sym(n_f-7*key_index,12)
            temp=x*np.exp(-1j*2*np.pi*(n_dr/12))*1j
            axs[2].plot(temp.real,temp.imag,'x',markersize=4,color=arrow_color)
            rho_DR += temp
    axs[0].quiver(0,0,rho_FR.real,rho_FR.imag, units='xy',scale=1,color=arrow_color)
    axs[1].quiver(0,0,rho_TR.real,rho_TR.imag, units='xy',scale=1,color=arrow_color)
    axs[2].quiver(0,0,rho_DR.real,rho_DR.imag, units='xy',scale=1,color=arrow_color)
        

def transformChroma(chroma):
    rho_FR = np.zeros_like(chroma,dtype=complex)
    rho_TR = np.zeros_like(chroma,dtype=complex)
    rho_DR = np.zeros_like(chroma,dtype=complex)
    chroma_index = getChromaticIndices()
    for time_index in range(chroma.shape[0]):
        for key_index,n_k in enumerate(num_accidentals):
            for chroma_i,chroma_bin in zip(chroma_index,chroma[time_index,:]):
                n_f = sym3(49*chroma_i,84,7*n_k)
                if n_f:
                    n_fr = sym(n_f-7*n_k, 48)
                if n_fr:
                    rho_FR[time_index,key_index] += chroma_bin*np.exp(-1j*2*np.pi*(n_fr/48))*1j        
                n_tr = sym(n_f-7*n_k-12,24)
                rho_TR[time_index,key_index] += chroma_bin*np.exp(-1j*2*np.pi*(n_tr/24))*1j           
                n_dr = sym(n_f-7*n_k,12)
                rho_DR[time_index,key_index] += chroma_bin*np.exp(-1j*2*np.pi*(n_dr/12))*1j
    keylabels = list(num_accidentals.values())
    return (keylabels,rho_FR,rho_TR,rho_DR)

if __name__=='__main__':
    fig,ax = plt.subplots(13,3,figsize=(5, 22),gridspec_kw={'width_ratios': [1, 1,1]})
    chroma = np.array([[1,0,0,0,1,0,0,1,0,0,0,0],[1,0,0,0,1,0,0,0,0,1,0,0]])
    plotCircleOfFifths(ax[0,1],False)
    #plotChromaVectorCircleF(ax[0,1],chroma[10,:])
    for i,n_k in enumerate(num_accidentals):    
        plotKeyRelatedRealPitches(ax[i+1,0],n_k,'fifths',False)
        plotKeyRelatedRealPitches(ax[i+1,1],n_k,'thirds',False)
        plotKeyRelatedRealPitches(ax[i+1,2],n_k,'diatonic',False)
        plotChromaVector(ax[i+1],chroma[0,:],n_k)
    for row in range(13):
        for col in range(3):
            ax[row,col].axis('off')

    plt.show()
    
