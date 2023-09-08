import matplotlib.pyplot as plt
import matplotlib
import librosa.display
import mir_eval
import itertools
import numpy as np

def plotChromagram(ax,t,chroma, downbeats=None,upbeats=None,vmax=0.5):
    img = librosa.display.specshow(chroma.T,x_coords=t.T,x_axis='time', y_axis='chroma', cmap="Reds", ax=ax, vmin=0, vmax=vmax)
    if downbeats is not None:
        downbeats = [beat for beat in downbeats if t[0] <= beat <= t[-1]]
        ax.vlines(downbeats,-0.5,11.5,'k',linestyles='dashed',alpha=0.6)
    if upbeats is not None:
        upbeats = [beat for beat in upbeats if t[0] <= beat <= t[-1]]
        ax.vlines(upbeats,-0.5,11.5,'k',linestyles='dashed',alpha=0.2)

    ax.set_xlabel("Time in s")
    return img

def getColor(chordlabel):
    colors = ["lightblue","blue", "green", "red", "orange", "purple", "grey", "lightgreen","brown", "magenta", "teal","cyan"]
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
    est_intervals.append([t_start,time[1]])
    est_labels.append(label[1])
    return np.array(est_intervals),est_labels

def chordTemplates(template_type="triads",noise=0):
    if template_type == "triads": # major, minor, augmented, diminished, suspended
        templates = np.zeros((6,12),dtype=float)
        labels = ["C","C:min","C:aug","C:dim","C:sus2","C:sus4"]
        for i,quality in enumerate(["maj","min","aug","dim","sus2","sus4"]):
            templates[i,:] = mir_eval.chord.quality_to_bitmap(quality) / 3
        return templates,labels
    elif template_type == "tetrads":
        templates = np.zeros((8,12),dtype=float)
        labels = ["C:7","C:maj7","C:min7","C:minmaj7","C:maj6","C:min6","C:dim7","C:hdim7"]
        for i,quality in enumerate(["7","maj7","min7","minmaj7","maj6","min6","dim7","hdim7"]):
            templates[i,:] = mir_eval.chord.quality_to_bitmap(quality) / 4
        return templates,labels
    
