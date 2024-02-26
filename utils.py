import librosa.display
import mir_eval
import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from collections import namedtuple
import pitchspace
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms


PitchClass = namedtuple('PitchClass','name pitch_class_index chromatic_index num_accidentals accident')
""" 
    pitch_class_index : index of pitch class in chroma vector and list of pitch_classes
    chromatic_index : index n_c for pitch class in pitch space 
    num_accidentals : The number of accidentals n_k present in the key of this pitch class 
"""

pitch_classes = [
            PitchClass("C",0,-2,0,""),
            PitchClass("Db",1,-1,-5,"b"), # use enharmonic note with lowest accidentals (Db)! (C# has 7 crosses) 
            PitchClass('D',2,0,2,"#"),
            PitchClass("Eb",3,1,-3,"b"), 
            PitchClass("E",4,2,4,"#"),
            PitchClass("F",5,3,-1,"b"),
            PitchClass("Gb",6,4,6,"b"),
            PitchClass("G",7,5,1,"#"),
            PitchClass("Ab",8,-6,-4,"b"),
            PitchClass("A",9,-5,3,"#"),
            PitchClass("Bb",10,-4,-2,"b"), 
            PitchClass("B",11,-3,5,"b")
]
"""A sorted list of Pitch classes: [C, C#/Db, .. , A#, B]"""

enharmonic_notes = {"C#":"Db","Db":"C#","D#":"Eb","Eb":"D#","F#":"Gb","Gb":"F#","G#":"Ab","Ab":"G#","A#":"Bb","Bb":"A#","B#":"C","C":"B#"}

def loadAudiofile(filepath,fs=22050,**kwargs):
    """load audio signal"""
    try:
        y,_ = librosa.load(filepath,mono=True,sr=fs,**kwargs)
        y = y / np.max(y) # normalize 
    except FileNotFoundError:
        raise FileNotFoundError(f"could not load file: {filepath}")
    return y

def loadChordAnnotations(annotationpath):
    try:
        intervals, labels = mir_eval.io.load_labeled_intervals(annotationpath)
    except FileNotFoundError:
        raise FileNotFoundError(f"Incorrect path! could not load annotation: {annotationpath}")
    return intervals, labels

def timeVector(N,t_start=0,t_stop=None,hop_length=512,sr=22050):
    if t_stop is None:
        time = np.arange(t_start,N * hop_length / sr,hop_length / sr)
    else:
        time = np.linspace(t_start,t_stop,N,endpoint=False)
    return time

def createChordIntervals(t,labels):
    """create chord intervals from a given sequence of labels by grouping them together+
    returns nd.array,chord_labels
    """
    est_labels = []
    est_intervals = []
    t_start = t[0]
    for time,label in zip(itertools.pairwise(t),itertools.pairwise(labels)):
        if(label[0] != label[1]): # chord change
            est_intervals.append([t_start,time[1]])
            est_labels.append(label[0])
            t_start = time[1]
        else: # no chord change
            continue
    if t_start != time[1]: # process last chord change
        est_intervals.append([t_start,time[-1]])
        est_labels.append(label[1])
    return np.array(est_intervals),est_labels

def evaluateTranscription(est_intervals,est_labels,ref_intervals,ref_labels,scheme="majmin"):
    """evaluate Transcription score according to MIREX scheme"""
    est_intervals, est_labels = mir_eval.util.adjust_intervals(
            est_intervals, est_labels, ref_intervals.min(),
            ref_intervals.max(), mir_eval.chord.NO_CHORD,
            mir_eval.chord.NO_CHORD)
    (intervals,ref_labels,est_labels) = mir_eval.util.merge_labeled_intervals(
        ref_intervals, ref_labels, est_intervals, est_labels)
    durations = mir_eval.util.intervals_to_durations(intervals)
    if scheme == "majmin":
        comparisons = mir_eval.chord.majmin(ref_labels, est_labels)
    elif scheme == "triads":
        comparisons = mir_eval.chord.triads(ref_labels, est_labels)
    elif scheme == "tetrads":
        comparisons = mir_eval.chord.tetrads(ref_labels, est_labels)
    elif scheme == "sevenths":
        comparisons = mir_eval.chord.sevenths(ref_labels, est_labels)
    else:
        raise ValueError(f"invalid evaluation scheme: {scheme}")
    score = round(mir_eval.chord.weighted_accuracy(comparisons, durations),2)
    mean_seg_score = round(mir_eval.chord.seg(ref_intervals, est_intervals),2)
    return score,mean_seg_score

def createChordTemplates(template_type="majmin"):
    """create a set of chord templates for the given evaluation scheme:

        majmin: "maj","min"
        triads: "maj","min","dim","aug"
        triads_extended: "maj","min","dim","aug","sus2","sus4"
        sevenths: "maj","min","maj7","7","min7"

        returns templates,chord_labels
    """
    if template_type == "majmin":
        quality = ["maj","min"]
    elif template_type == "triads":
        quality = ["maj","min","dim","aug"]
    elif template_type == "triads_extended":    
        quality = ["maj","min","dim","aug","sus2","sus4"]
    elif template_type == "sevenths":
        quality = ["maj","min","maj7","7","min7"]
    elif template_type == "triads_tetrads":
        quality = ["maj","min","dim","aug","sus2","sus4","maj7","7","min7","hdim7"]
    templates = np.zeros((12,12*len(quality)+1),dtype=float)
    chord_labels = []
    chord_index = 0

    # create chord templates + No chord prototype
    for q in quality:  
        template = mir_eval.chord.quality_to_bitmap(q)
        template = template / np.linalg.norm(template)
        for pitch in pitch_classes:
            templates[:,chord_index] = np.roll(template,pitch.pitch_class_index)
            chord_labels.append(f"{pitch.name}:{q}")
            chord_index += 1
    for i in range(0,12,2):
        templates[i,chord_index] = 1/6

    chord_labels.append("N")
    return templates,chord_labels

def createKeyTemplates():
    key_templates = np.zeros((12,12),dtype=float)
    for i in range(12):
        key_templates[:,i] = np.roll([1,0,1,0,1,1,0,1,0,1,0,1],shift=i)
    return key_templates

def getTimeIndices(timevector,time_interval):
    i0 = 0
    i1 = -1
    if time_interval is not None:
        if timevector.size > 0 and time_interval[0] <= timevector[-1]:
            i0 = np.searchsorted(timevector, time_interval[0], side='left')
            i1 = np.searchsorted(timevector, time_interval[1], side='left')

            # Ensure i1 is within the bounds of the array
            i1 = min(i1, timevector.shape[0] - 1)
    return i0,i1

def plotChromagram(ax,t_chroma,chroma,time_interval=None):
    i0,i1 = getTimeIndices(t_chroma,time_interval)
    img = librosa.display.specshow(chroma[:,i0:i1],x_coords=t_chroma[i0:i1],x_axis="time", y_axis='chroma', cmap="Reds", ax=ax,vmin=0, vmax=np.max(chroma[:,i0:i1]))
    return img

def plotCQT(ax,t_chroma,cqt,time_interval=None):
    i0,i1 = getTimeIndices(t_chroma,time_interval)
    img = librosa.display.specshow(cqt[:,i0:i1],
                                x_coords=t_chroma[i0:i1],
                                x_axis="time",
                                y_axis='cqt_note',
                                cmap="viridis",
                                ax=ax,vmin=0,
                                vmax=np.max(cqt[:,i0:i1]))
    return img

def plotCorrelation(ax,t_chroma,correlation,time_interval=None):
    i0,i1 = getTimeIndices(t_chroma,time_interval)
    img = librosa.display.specshow(correlation[:,i0:i1],x_coords=t_chroma[i0:i1],x_axis="time", y_axis='cqt', cmap="viridis", ax=ax,vmin=0, vmax=np.max(correlation[:,i0:i1]))
    pass

def plotChordAnnotations(ax,intervals,labels,time_interval=(0,10),y_0=0):
    def getColor(chordlabel):
        colors = ["lightblue","blue", "green", "red", "orange", "purple", "grey", "lightgreen","brown", "magenta", "teal","cyan","white"]
        root,_,_ = mir_eval.chord.encode(chordlabel)
        return colors[root]
    for i,label in enumerate(labels):
        # skip labels that do not overlap time interval 
        if intervals[i,1] < time_interval[0] or intervals[i,0] > time_interval[1]:
            continue
        # set start position of rectangular patch
        t_start = max(intervals[i,0],time_interval[0])
        t_stop = min(intervals[i,1],time_interval[1])
        rect = Rectangle((t_start, y_0),t_stop - t_start , 1.4, linewidth=1, edgecolor="k", facecolor=getColor(label))
        ax.add_patch(rect)
        if t_stop - t_start > 0.5:
            ax.text(t_start+ (t_stop - t_start)/2, y_0 + 0.6, label,verticalalignment="center",horizontalalignment='center', fontsize=9, color='k')
    ax.set_xlim(time_interval)

def plotSSM(ax,time_vector,S,time_interval=None,cmap="viridis"):
    if time_interval is not None:
        index_start = np.where(time_vector >= time_interval[0])[0][0]
        index_stop = np.where(time_vector > time_interval[1])[0][0]
    else:
        index_start = 0
        index_stop = time_vector.shape[0]
    ticks = np.linspace(0,index_stop-index_start,5)
    ticklabels = np.linspace(time_vector[index_start],time_vector[index_stop],5,dtype=int)
    img = librosa.display.specshow(S[index_start:index_stop,index_start:index_stop],ax=ax,cmap=cmap,vmax=1)
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticklabels)
    ax.set_yticks(ticks)
    ax.set_yticklabels(ticklabels)
    ax.set_xlabel("Time in s")
    ax.set_ylabel("Time in s")
    return img

def plotRecurrencePlots(t_chroma,W,SSM,SSM_M):
    fig,(ax0,ax1,ax2) = plt.subplots(1,3,figsize=(9,2.6))
    plotSSM(ax0,t_chroma,W,time_interval=(0,30))
    plotSSM(ax1,t_chroma,SSM,time_interval=(0,30))
    plotSSM(ax2,t_chroma,SSM_M,time_interval=(0,30))
    return fig

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
    
def drawLabel(axis,z,note_label,radius = 1.25,fontsize=8):
    x,y= scaleVector(z.real,z.imag,radius)
    axis.text(x,y,note_label,rotation=np.arctan2(y,x)*180/np.pi-90,
              fontsize=fontsize,horizontalalignment='center',verticalalignment='center')

def plotHalftoneGrid(axis,n_ht):
    axis.plot([-1,1], [0,0], color='grey',linestyle=(0,(5,10)),linewidth=1)
    axis.plot([0,0], [-1,1], color='grey',linestyle=(0,(5,10)),linewidth=1)
    for i in np.linspace(0,n_ht,n_ht+1):
        ht = np.exp(-1j*2*np.pi*(i/n_ht))*1j
        axis.plot(ht.real,ht.imag,'o',color='grey',markersize=1.5)

def plotCircleF(ax):
    plotHalftoneGrid(ax, 84)
    for pitch_class in pitch_classes:
        n_f = pitchspace.sym3(49 * pitch_class.chromatic_index, 84, 0)
        rho_F = np.exp(-1j * 2 * np.pi * (n_f / 84)) * 1j
        ax.plot(rho_F.real, rho_F.imag, 'o', color='k', markersize=3)
        drawLabel(ax, rho_F, pitch_class.name)
    ax.axis('off')

def plotCircleFR(ax,pitch_class_index=0):
    plotHalftoneGrid(ax,48)
    n_k = pitch_classes[pitch_class_index].num_accidentals
    ax.axline((0, -.8), (0, .8),linestyle="--",color="grey",alpha=0.5)
    for pitch_class in pitch_classes:
        n_f = pitchspace.sym3(49*pitch_class.chromatic_index,84,7*n_k)
        # check if pitch is in fifth related circle
        if -21 <= (n_f-7*n_k) <= 21:
            n_fr = pitchspace.sym(n_f-7*n_k, 48)
            rho_FR = np.exp(-1j*2*np.pi*(n_fr/48))*1j
            ax.plot(rho_FR.real,rho_FR.imag,'ok',markersize=3)
            # check if the current key has #'s in it and use enharmonic notes to annotate the correct note
            if pitch_class.name.find("b") == 1 and pitch_classes[pitch_class_index].accident == "#":
                drawLabel(ax,rho_FR,enharmonic_notes[pitch_class.name])
            else:
                drawLabel(ax,rho_FR,pitch_class.name)
    ax.axis('off')

def plotCircleTR(ax,pitch_class_index=0):
    plotHalftoneGrid(ax,24)
    n_k = pitch_classes[pitch_class_index].num_accidentals
    for pitch_class in pitch_classes:
        n_f = pitchspace.sym3(49*pitch_class.chromatic_index,84,7*n_k)
        # check if pitch is in fifth related circle
        if -21 <= (n_f-7*n_k) <= 21:
            n_tr = pitchspace.sym(n_f-7*n_k-12,24)
            rho_TR = np.exp(-1j*2*np.pi*((n_tr)/24))*1j
            ax.plot(rho_TR.real,rho_TR.imag,'ok',markersize=3)
            if pitch_class.name.find("b") == 1 and pitch_classes[pitch_class_index].accident == "#":
                drawLabel(ax,rho_TR,enharmonic_notes[pitch_class.name])
            else:
                drawLabel(ax,rho_TR,pitch_class.name)

    ax.axline((0, -.9), (0, .9),linestyle="--",color="grey",alpha=0.5)
    ax.axis('off')

def plotCircleDR(ax,pitch_class_index=0):
    plotHalftoneGrid(ax,12)
    n_k = pitch_classes[pitch_class_index].num_accidentals
    # ax.text(-1.1,1.2,f"DR({pitch_classes[pitch_class_index].name})",fontsize=8,
    #           horizontalalignment='center',verticalalignment='center')
    for pitch_class in pitch_classes:
        n_f = pitchspace.sym3(49*pitch_class.chromatic_index,84,7*n_k)
        # check if pitch is in fifth related circle
        if -21 <= (n_f-7*n_k) <= 21:
            n_dr = pitchspace.sym(n_f-7*n_k,12)
            rho_DR = np.exp(-1j*2*np.pi*(n_dr/12))*1j
            ax.plot(rho_DR.real,rho_DR.imag,'ok',markersize=3)
            if pitch_class.name.find("b") == 1 and pitch_classes[pitch_class_index].accident == "#":
                drawLabel(ax,rho_DR,enharmonic_notes[pitch_class.name])
            else:
                drawLabel(ax,rho_DR,pitch_class.name)
  
    ax.axline((0, -.9), (0, .9),linestyle="--",color="grey",alpha=0.5)
    ax.axis('off')

def plotPitchspace():
    n=12
    fig,ax = plt.subplots(nrows=n, ncols=3, figsize=(3.3,n*1+0.3))
    ax[0, 0].text(0,2,f"FR",ha='center',va='center',fontsize=12)
    ax[0, 1].text(0,2,f"TR",ha='center',va='center',fontsize=12)
    ax[0, 2].text(0,2,f"DR",ha='center',va='center',fontsize=12)
    for i in range(n):
        plotCircleFR(ax[i, 0], i)
        plotCircleTR(ax[i, 1], i)
        plotCircleDR(ax[i, 2], i)
        ax[i, 0].text(-2,0,f"{pitch_classes[i].name}",ha='center',va='center',fontsize=12)

    fig.subplots_adjust(left=0.1, top=0.9)
    fig.tight_layout(w_pad=0.5,h_pad=1)    
    return fig,ax

def create_violinplot(ax,data,xlabels,bodycolor='cyan'):
    violin_parts = ax.violinplot(data,showmeans=False, showmedians=True,
            showextrema=False)
    violin_parts["cmedians"].set_color("red")
    for x in violin_parts["bodies"]:
        x.set_color(bodycolor)
    bplot_parts = ax.boxplot(data,
                showfliers=True,medianprops=dict(linestyle=None,linewidth=0),
                flierprops=dict(markerfacecolor='k', marker='o',markersize=1),
                widths=0.1)
    ax.set_yticks(np.arange(0,110,10))
    ax.set_ylim(0,100)
    ax.set_ylabel("F-score in %")
    ax.set_xticks(np.arange(1, len(xlabels) + 1), labels=xlabels)
    ax.set_xlim(0.5, len(xlabels) + 0.5);
    ax.grid("on")
    return violin_parts, bplot_parts

def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    https://matplotlib.org/stable/gallery/statistics/confidence_ellipse.html

    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y,ddof=0)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)
