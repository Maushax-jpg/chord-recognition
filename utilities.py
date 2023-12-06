import matplotlib.pyplot as plt
import matplotlib
import librosa.display
import mir_eval
import itertools
import numpy as np
import madmom
import pitchspace
import os.path
from collections import namedtuple

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


def loadAudio(audiopath,t_start=0,t_stop=None,fs=22050,hpss=None):
    """wrapper function that loads audio file using the madmom Signal class"""
    if hpss is not None:
        basepath,filename = os.path.split(audiopath)
        filename = filename.rsplit('.', 1)[0]
        if hpss == "harmonic":
            audiopath = os.path.join(basepath,f"{filename}_harmonic.mp3")
        elif hpss == "harmonic+drums":
            audiopath = os.path.join(basepath,f"{filename}_harmonic+drums.mp3")
        elif hpss == "harmonic+vocals":
            audiopath = os.path.join(basepath,f"{filename}_harmonic+vocals.mp3")
        else:
            raise ValueError(f"Invalid separation argument: {hpss}")
    try:
        y, _ = madmom.io.audio.load_audio_file(audiopath, sample_rate=fs, num_channels=1, start=t_start, stop=t_stop, dtype=float)
    except FileNotFoundError:
        raise ValueError(audiopath)
    sig = madmom.audio.signal.Signal(y, sample_rate=fs, num_channels=1, start=t_start, stop=t_stop)
    sig = madmom.audio.signal.normalize(sig)
    timevector = np.linspace(sig.start,sig.stop,sig.num_samples)
    return timevector,sig

def loadAnnotations(annotationpath):
    intervals, labels = mir_eval.io.load_labeled_intervals(annotationpath)
    return intervals, labels

def plotChromagram(ax,t,chroma, downbeats=None,upbeats=None,chroma_type="crp",time="seconds"):
    if time == "seconds":
        x_coords = t
        x_axis = "time"
    else:
        x_coords = None
        x_axis = None
        
    if chroma_type=="crp":
        img = librosa.display.specshow(chroma,x_coords=x_coords,x_axis=x_axis, y_axis='chroma', cmap="bwr", ax=ax,vmin=-np.max(chroma), vmax=np.max(chroma))
    elif chroma_type=="log":
        img = librosa.display.specshow(chroma,x_coords=x_coords,x_axis=x_axis, y_axis='chroma', cmap="Reds", ax=ax,vmin=np.min(chroma), vmax=np.max(chroma))
    else:
        img = librosa.display.specshow(chroma,x_coords=x_coords,x_axis=x_axis, y_axis='chroma', cmap="Reds", ax=ax,vmin=0, vmax=0.5)
    
    if downbeats is not None:
        downbeats = [beat for beat in downbeats if t[0] <= beat <= t[-1]]
        ax.vlines(downbeats,-0.5,11.5,'k',linestyles='dashed',alpha=0.6)
    if upbeats is not None:
        upbeats = [beat for beat in upbeats if t[0] <= beat <= t[-1]]
        ax.vlines(upbeats,-0.5,11.5,'k',linestyles='dashed',alpha=0.2)

    ax.set_xlabel("Time in s")
    return img

def smoothChromagram(t,chroma,beats):
    chroma_smoothed = np.copy(chroma)
    for b0,b1 in itertools.pairwise(beats):
        # median filter
        try:
            idx0 = np.argwhere(t >= b0)[0][0]
        except IndexError:
            # no matching interval found at array boundaries
            idx1 = 0
        try:
            idx1 = np.argwhere(t >= b1)[0][0]
        except IndexError:
            idx1 = t.shape[0] 
        if idx1-idx0 > 0: 
            chroma_mean = np.mean(chroma[idx0:idx1,:],axis=0)
            chroma_smoothed[idx0:idx1,:] = np.tile(chroma_mean,(idx1-idx0,1))
    return chroma_smoothed

def getColor(chordlabel):
    colors = ["lightblue","blue", "green", "red", "orange", "purple", "grey", "lightgreen","brown", "magenta", "teal","cyan","white"]
    root,_,_ = mir_eval.chord.encode(chordlabel)
    return colors[root]

def formatChordLabel(chordlabel):
    # latex formatting of chordlabels, ignore additional scale degrees
    root,quality,additional_notes,bass = mir_eval.chord.split(chordlabel)
    if root[-1] == '#':
        root = root[:-1] + "\#"
    elif root[-1] == 'b':
        root = root[:-1] + "\\flat "
    if quality == 'min':
        quality = "m"
    elif quality == "maj":
        quality = ""
    if quality is None:
        return f"${root}$"
    else:
        return f"${root}{quality}$"

def plotEstimatedKeys(ax,t,est_keys):   
    """not implemented yet"""
    # maybe plot similar to chroma? not sure yet 
    key_matrix = np.zeros((t.shape[0],12),dtype=float)
    for i,key_candidates in enumerate(est_keys):
        for n,key in enumerate(key_candidates):
            key_matrix[i,key[0]] = key[1]
    plotChromagram(ax,t,key_matrix,vmax=0.2)

def plotChordAnnotations(ax,annotations,time_interval=(0,10),format_label=False):
    ref_intervals, ref_labels = annotations
    for i,label in enumerate(ref_labels):
        # skip labels that do not overlap time interval 
        if ref_intervals[i,1] < time_interval[0] or ref_intervals[i,0] > time_interval[1]:
            continue
        # set start position of rectangular patch
        t_start = max(ref_intervals[i,0],time_interval[0])
        t_stop = min(ref_intervals[i,1],time_interval[1])

        rect = matplotlib.patches.Rectangle((t_start, 0),t_stop - t_start , 1.4, linewidth=1, edgecolor="k", facecolor=getColor(label))
        ax.add_patch(rect)
        if format_label:
            label = formatChordLabel(label)
        if t_stop - t_start > 0.4:
            ax.text(t_start+ (t_stop - t_start)/2, 0.6, label,verticalalignment="center",horizontalalignment='center', fontsize=9, color='k')
    ax.set_ylim(0,2)
    ax.axis("off")
    ax.set_xlim(time_interval)

def plotIntervalCategories(ax,t,interval_categories,ax_colorbar=None):
    img = librosa.display.specshow(interval_categories.T,x_coords=t.T,x_axis='time', cmap="Reds", ax=ax, vmin=0, vmax=0.3)
    ax.set_yticks(np.arange(6))
    ax.set_yticklabels(["IC1","IC2","IC3","IC4","IC5","IC6"]);
    ax.set_xlim(t[0],t[-1])
    ax.set_xticklabels([])
    if ax_colorbar is not None:
        fig = plt.get_current_fig_manager()
        fig.colorbar(img,cax=ax,cmap="Reds")

def plotComplexityFeatures(ax,ax_label,t,features):
    colors = ["blue", "green", "red", "orange", "purple", "grey","brown", "black"]
    ax.plot(t, features[0],color=colors[0])
    ax.plot(t, features[1],color=colors[1])
    ax.plot(t, features[2],color=colors[2])
    ax.plot(t, features[3],color=colors[3])
    ax.plot(t, features[4],color=colors[4])
    ax.plot(t, features[5],color=colors[5])
    ax.plot(t, features[6],color=colors[6])
    ax.set_xlim(t[0],t[-1])
    ax.set_ylim(0,1)
    ax.grid(True)
    #create legend
    labels = ['Sum of diff.',"Fifth Width","Flatness","Entropy","Linear Slope","Non-Sparseness","std"]
    legend_handles = [matplotlib.patches.Patch(color=color) for color in colors]
    legend = ax_label.legend(legend_handles, labels, loc='center left', bbox_to_anchor=(0, 0.5), title='complexity features', handlelength=1, handletextpad=0.5, fontsize=8, title_fontsize=10, facecolor='lightgray', framealpha=0.8)
    ax_label.add_artist(legend)
    ax_label.set_axis_off()

def createChordIntervals(t,labels):
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

def createChordTemplates(template_type="majmin"):
    """create a set of chord templates for the given evaluation scheme:

        majmin: "maj","min"
        triads: "maj","min","dim","aug"
        triads_extended: "maj","min","dim","aug","sus2","sus4"
        majmin_sevenths: "maj","min","maj7","7","min7"

        returns templates,chord_labels
    """
    if template_type == "majmin":
        quality = ["maj","min"]
    elif template_type == "triads":
        quality = ["maj","min","dim","aug"]
    elif template_type == "triads_extended":    
        quality = ["maj","min","dim","aug","sus2","sus4"]
    elif template_type == "majmin_sevenths":
        quality = ["maj","min","maj7","7","min7"]
    templates = np.zeros((12,12*len(quality)+1),dtype=float)
    chord_labels = []
    chord_index = 0

    # create chord templates + No chord prototype
    for q in quality:  
        template = mir_eval.chord.quality_to_bitmap(q)
        template = template / np.sum(template)
        for pitch in pitchspace.pitch_classes:
            templates[:,chord_index] = np.roll(template,pitch.pitch_class_index)
            chord_labels.append(f"{pitch.name}:{q}")
            chord_index += 1
    for i in range(12):
        templates[i,chord_index] = -1/12
    chord_labels.append("N")
    return templates,chord_labels

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

def saveTranscriptionResults(output_path,name,t_chroma,chroma,est_intervals,est_labels,ref_intervals,ref_labels):
    # store transcription and save a preview
    if output_path is not None:
        fpath = f"{output_path}/transcriptions/{name}.chords"
        f = open(fpath, "w")
        # save to a .chords file
        for interval,label in zip(est_intervals,est_labels):
            f.write(f"{interval[0]:0.6f}\t{interval[1]:0.6f}\t{label}\n")
        f.close()

        fig,ax = plt.subplots(3,2,height_ratios=(1,1,10),width_ratios=(9.5,.5),figsize=(15,10))
        plotChordAnnotations(ax[0,0],(ref_intervals,ref_labels),(0,15))
        ax[0,0].text(0,1.7,"Annotated chords")
        plotChordAnnotations(ax[1,0],(est_intervals,est_labels),(0,15))
        ax[1,0].text(0,1.7,"Estimated chords")
        ax[1,1].set_axis_off()
        ax[0,1].set_axis_off()
        img = plotChromagram(ax[2,0],t_chroma,chroma,None,None,vmin=-np.max(chroma),vmax=np.max(chroma),cmap='bwr')
        fig.colorbar(img,cax=ax[2,1],cmap="bwr")
        ax[2,0].set_xlim([0,15])
        plt.savefig(f"{output_path}/chromagrams/{name}.pdf", dpi=600)
        plt.close()
    
def plotHalftoneGrid(axis,n_ht):
    axis.plot([-1,1], [0,0], color='grey',linestyle=(0,(5,10)),linewidth=1)
    axis.plot([0,0], [-1,1], color='grey',linestyle=(0,(5,10)),linewidth=1)
    for i in np.linspace(0,n_ht,n_ht+1):
        ht = np.exp(-1j*2*np.pi*(i/n_ht))*1j
        axis.plot(ht.real,ht.imag,'o',color='grey',markersize=1.5)

def drawLabel(axis,z,note_label,radius = 1.15,fontsize=7):
    x = z.real * radius
    y = z.imag * radius
    axis.text(x,y,note_label,rotation=np.arctan2(y,x)*180/np.pi-90,
              fontsize=fontsize,horizontalalignment='center',verticalalignment='center')

def plotCircleF(ax):
    plotHalftoneGrid(ax,84)
    for pitch_class in pitch_classes:
        n_f = pitchspace.sym3(49*pitch_class.chromatic_index,84,0)
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
    ax.axline((0, -.8), (0, .8),linestyle="--",color="grey",alpha=0.5)
    for pitch_class in pitch_classes:
        n_f = pitchspace.sym3(49*pitch_class.chromatic_index,84,7*n_k)
        # check if pitch is in fifth related circle
        if -21 <= (n_f-7*n_k) <= 21:
            n_fr = pitchspace.sym(n_f-7*n_k, 48)
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
        n_f = pitchspace.sym3(49*pitch_class.chromatic_index,84,7*n_k)
        # check if pitch is in fifth related circle
        if -21 <= (n_f-7*n_k) <= 21:
            n_tr = pitchspace.sym(n_f-7*n_k-12,24)
            rho_TR = np.exp(-1j*2*np.pi*((n_tr)/24))*1j
            ax.plot(rho_TR.real,rho_TR.imag,'ok',markersize=3)
            drawLabel(ax,rho_TR,pitch_class.name)
            continue
        if alterations:
            n_f_sharp = pitchspace.sym3(49*pitch_class.chromatic_index,84,7*n_k_sharp)
            n_tr_sharp = pitchspace.sym(n_f_sharp-7*n_k-12,24)
            r_tr_sharp = np.exp(-1j*2*np.pi*((n_tr_sharp)/24))*1j
            drawLabel(ax,r_tr_sharp,pitch_class.name)
    ax.axline((0, -.9), (0, .9),linestyle="--",color="grey",alpha=0.5)
    ax.axis('off')

def plotCircleDR(ax,pitch_class_index=0):
    plotHalftoneGrid(ax,12)
    n_k = pitch_classes[pitch_class_index].num_accidentals
    ax.text(-1.1,1.2,f"DR({pitch_classes[pitch_class_index].name})",fontsize=8,
              horizontalalignment='center',verticalalignment='center')
    for pitch_class in pitch_classes:
        n_f = pitchspace.sym3(49*pitch_class.chromatic_index,84,7*n_k)
        # check if pitch is in fifth related circle
        if -21 <= (n_f-7*n_k) <= 21:
            n_dr = pitchspace.sym(n_f-7*n_k,12)
            rho_DR = np.exp(-1j*2*np.pi*(n_dr/12))*1j
            ax.plot(rho_DR.real,rho_DR.imag,'ok',markersize=3)
            drawLabel(ax,rho_DR,pitch_class.name)          
    ax.axline((0, -.9), (0, .9),linestyle="--",color="grey",alpha=0.5)
    ax.axis('off')