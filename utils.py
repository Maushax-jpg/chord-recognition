import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import mir_eval
import utils

def buffer(X, n, overlap ,n_zeros ,window : np.ndarray) -> np.ndarray:
    # buffers data vector X into length n column vectors with overlap p
    # and zero padding on both ends of the signal
    # excess data at the end of X is discarded
    L = len(X)  # length of data to be buffered
    m = int(np.floor((L - n) / (n - overlap)) + 1)  # number of sample vectors (no padding)
    data = np.zeros((n+n_zeros, m))  # initialize data matrix
    startIndex = n_zeros // 2
    if n <= 0 or overlap < 0 or overlap >= n:
        raise ValueError("Invalid buffer parameters")
    for i, column in zip(range(0, L - n + 1, n - overlap), range(0, m)):
        data[startIndex : startIndex + n, column] = X[i : i + n] * window
    return data

def getColor(chordlabel):
    colors = ["red", "green", "lightblue", "yellow", "orange", "purple", "pink", "cyan", "brown", "magenta", "teal", "gray","white"]
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

def plotChromagram(ax,t,chroma, beats=None,downbeats=True):
    img = librosa.display.specshow(chroma.T,x_coords=t.T,x_axis='time', y_axis='chroma', cmap="Reds", ax=ax, vmin=0, vmax=0.5)
    if beats is not None:
        if downbeats:
            for beat,downbeat in beats:
                if downbeat==1:
                    alpha=0.6
                else:
                    alpha=0.2
                ax.vlines(beat,-0.5,11.5,'k',linestyles='dashed',alpha=alpha)
        else:
            ax.vlines(beats,-0.5,11.5,'k',linestyles='dashed',alpha=0.2)
    return img
    

def plotChordAnnotations(ax,annotations,time_interval=(0,10)):
    ref_intervals, ref_labels = annotations
    colors = ["red", "green", "blue", "yellow", "orange", "purple", "pink", "cyan", "brown", "magenta", "teal", "gray"]
    for i,label in enumerate(ref_labels):
        # skip labels that do not overlap time interval 
        if ref_intervals[i,1] < time_interval[0] or ref_intervals[i,0] > time_interval[1]:
            continue
        # set start position of rectangular patch
        t_start = max(ref_intervals[i,0],time_interval[0])
        t_stop = min(ref_intervals[i,1],time_interval[1])

        rect = patches.Rectangle((t_start, 0),t_stop - t_start , 1.4, linewidth=1, edgecolor="k", facecolor=getColor(label))
        ax.add_patch(rect)
        ax.text(t_start+ (t_stop - t_start)/2, 0.6, formatChordLabel(label),verticalalignment="center",horizontalalignment='center', fontsize=10, color='k')
    ax.set_ylim(0,2)
    ax.axis("off")
    ax.set_xlim(time_interval)
    
def plotCqt(ax,signal):
    cqt = librosa.cqt(signal,hop_length=4410,sr=44100)
    librosa.display.specshow(librosa.amplitude_to_db(np.abs(cqt), ref=np.max),y_axis='cqt_note',cmap='viridis',
                                ax=ax,x_axis='time',sr=44100,hop_length=4410)
                                
def plotAudioWaveform(ax,y,time_interval=(0,10),beats=None,downbeats=False):
    time_vector = np.linspace(time_interval[0], time_interval[1], y.shape[0], endpoint=False)
    ax.plot(time_vector,y)
    if beats is not None:
        if downbeats:
            for beat,downbeat in beats:
                if beat < time_interval[0]:
                    continue
                elif beat > time_interval[1]:
                    break
                if downbeat==1:
                    alpha=0.8
                else:
                    alpha=0.2
                ax.vlines(beat,-1,1,'k',linestyles='dashed',alpha=alpha)
    ax.set_ylim(-1,1)
    ax.set_xlim(time_interval)

def plotFeatures(start,stop,time_vector,y,chroma,beats,complexity_features,intervals,target,estimation_cps,estimation_templates):

        # define gridspec and axes
    fig = plt.figure(figsize=(14*0.8, 10*0.8))
    gs = gridspec.GridSpec(8, 3,width_ratios=(40,1,5),height_ratios=(1,1,1,1,6,6,4,2), hspace=0.2,wspace=0.05)
    ax00 = fig.add_subplot(gs[0,0])
    ax01 = fig.add_subplot(gs[0,1:3])
    ax10 = fig.add_subplot(gs[1,0])
    ax11 = fig.add_subplot(gs[1,1:3])
    ax20 = fig.add_subplot(gs[2,0])
    ax21 = fig.add_subplot(gs[2,1:3])
    ax30 = fig.add_subplot(gs[3,0])

    ax40 = fig.add_subplot(gs[4,0])
    ax41 = fig.add_subplot(gs[4,1])
    ax50 = fig.add_subplot(gs[5,0])
    ax51 = fig.add_subplot(gs[5,1:3])
    ax60 = fig.add_subplot(gs[6,0])
    ax61 = fig.add_subplot(gs[6,1])
    ax70 = fig.add_subplot(gs[7,0])

    utils.plotChordAnnotations(ax00,target,time_interval=(start,stop))
    ax10.set_xlim(time_vector[0],time_vector[-1])
    ax01.text(0,0,"Annotation",fontsize=11)
    ax01.set_axis_off()

    utils.plotChordAnnotations(ax10,estimation_cps,time_interval=(start,stop))
    ax10.set_xlim(time_vector[0],time_vector[-1])
    ax11.text(0,0,"CPS",fontsize=11)
    ax11.set_axis_off()

    utils.plotChordAnnotations(ax20,estimation_templates,time_interval=(start,stop))
    ax20.set_xlim(time_vector[0],time_vector[-1])
    ax21.text(0,0,"Templates",fontsize=11)
    ax21.set_axis_off()

    utils.plotAudioWaveform(ax30,y,time_interval=(start,stop),beats=beats)
    ax30.set_xlim(time_vector[0],time_vector[-1])
    ax30.set_axis_off()

    img = utils.plotChromagram(ax40, time_vector, chroma,beats=beats)
    ax40.set_xlabel("")
    ax40.set_xlim(time_vector[0],time_vector[-1])
    ax40.set_xticklabels([])
    fig.colorbar(img,cax=ax41,cmap="Reds")

    # features
    colors = ["blue", "green", "red", "orange", "purple", "grey", "black"]
    for i,feature in enumerate(complexity_features):
        ax50.plot(time_vector, feature,color=colors[i])
    ax50.set_xlim(time_vector[0],time_vector[-1])
    ax50.set_xticklabels([])
    ax50.set_ylim(0,1)
    ax50.grid(True)
    #create legend
    labels = ['Sum of diff.',"Fifth Width","Flatness","Entropy","Linear Slope","Non-Sparseness"]
    legend_handles = [patches.Patch(color=color) for color in colors]
    legend = ax51.legend(legend_handles, labels, loc='center left', bbox_to_anchor=(0, 0.5), title='complexity features', handlelength=1, handletextpad=0.5, fontsize=8, title_fontsize=10, facecolor='lightgray', framealpha=0.8)
    ax51.add_artist(legend)
    ax51.set_axis_off()

    img = librosa.display.specshow(intervals.T,x_coords=time_vector.T,x_axis='time', cmap="Reds", ax=ax60, vmin=0, vmax=0.3)
    ax60.set_yticks(np.arange(6))
    ax60.set_yticklabels(["IC1","IC2","IC3","IC4","IC5","IC6"]);
    ax60.set_xlim(time_vector[0],time_vector[-1])
    ax60.set_xticklabels([])
    fig.colorbar(img,cax=ax61,cmap="Reds")

    ax70.plot(time_vector,np.sum(np.array(complexity_features),axis=0))
    ax70.set_ylabel("complexity")
    ax70.set_xlim(time_vector[0],time_vector[-1]);
    ax70.set_xlabel("Time in s")

    return fig