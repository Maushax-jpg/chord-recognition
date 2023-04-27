import numpy as np
import matplotlib.pyplot as plt

# chromatic index of a pitch
# denoted as n_i
chromatic_index = {-6:'gis',-5:'a',-4:'b',-3:'h',-2:'c',-1:'cis',0:'d',1:'es',2:'e',3:'f',4:'fis',5:'g'}

# number of accidentals: flats (+) or sharps (-) in a key  (e.g. F-maj=-1,C-maj=0, G-Maj=1)
# denoted as n_k
num_accidentals = {-5:'Des',-4:'As',-3:'Es',-2:'B',-1:'F',0:'C',1:'G',2:'D',3:'A',4:'E',5:'H',6:'Fis'}

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
    x,y= scaleVector(z.real,z.imag,1.2)
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

def plotCircleOfFifths(axis,chroma=None):
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
    # cartesian 2 polar conversion
    rho_F = list(map(lambda x: cartesian2polar(x.real,x.imag), rho_F))
    rho_FR = list(map(lambda fr: [cartesian2polar(x.real,x.imag) for x in fr], rho_FR))
    rho_TR = list(map(lambda tr: [cartesian2polar(x.real,x.imag) for x in tr], rho_TR))
    rho_DR = list(map(lambda dr: [cartesian2polar(x.real,x.imag) for x in dr], rho_DR))
    return (rho_F,rho_FR,rho_TR,rho_DR)

def plotPitchSpace(size="A4"):
    if size == 'A4':
        fig_cps = plt.figure(figsize=((8.27,11.69)))
    else:
        # other formats?
        return    
    # helper variable
    n_k = list(num_accidentals.keys())
    grid = plt.GridSpec(7, 7, wspace=0.5, hspace=0.5)
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
    fig,ax = plt.subplots(figsize=(5, 5))
    chroma = np.array([[1,0,0,0,1,0,0,1,0,0,0,0],[1,0,0,0,1,0,0,0,0,1,0,0]])
    plotCircleOfFifths(ax,False)
    #plotChromaVectorCircleF(ax[0,1],chroma[10,:])
    # for i,n_k in enumerate(num_accidentals):    
    #     plotKeyRelatedRealPitches(ax[i+1,0],n_k,'fifths',False)
    #     plotKeyRelatedRealPitches(ax[i+1,1],n_k,'thirds',False)
    #     plotKeyRelatedRealPitches(ax[i+1,2],n_k,'diatonic',False)
    #     plotChromaVector(ax[i+1],chroma[0,:],n_k)
    # for row in range(13):
    #     for col in range(3):
    #         ax[row,col].axis('off')

    plt.show()
    
