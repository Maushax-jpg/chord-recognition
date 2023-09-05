import matplotlib.pyplot as plt
import matplotlib
import librosa.display
import mir_eval
import itertools
import numpy as np

def plotChromagram(ax,t,chroma, downbeats=None,upbeats=None):
    img = librosa.display.specshow(chroma.T,x_coords=t.T,x_axis='time', y_axis='chroma', cmap="Reds", ax=ax, vmin=0, vmax=0.5)
    if downbeats is not None:
        downbeats = [beat for beat in downbeats if t[0] <= beat <= t[-1]]
        ax.vlines(downbeats,-0.5,11.5,'k',linestyles='dashed',alpha=0.6)
    if upbeats is not None:
        upbeats = [beat for beat in upbeats if t[0] <= beat <= t[-1]]
        ax.vlines(upbeats,-0.5,11.5,'k',linestyles='dashed',alpha=0.2)
    return img

def getColor(chordlabel):
    colors = ["blue","lightblue", "green", "red", "orange", "purple", "grey", "black","brown", "magenta", "teal","cyan"]
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
    

def plotChordAnnotations(ax,annotations,time_interval=(0,10),format_label=False):
    ref_intervals, ref_labels = annotations
    colors = ["red", "green", "blue", "yellow", "orange", "purple", "pink", "cyan", "brown", "magenta", "teal", "gray"]
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
        ax.text(t_start+ (t_stop - t_start)/2, 0.6, label,verticalalignment="center",horizontalalignment='center', fontsize=10, color='k')
    ax.set_ylim(0,2)
    ax.axis("off")
    ax.set_xlim(time_interval)
    
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
    est_intervals.append([t_start,time[1]])
    est_labels.append(label[1])
    return np.array(est_intervals),est_labels

