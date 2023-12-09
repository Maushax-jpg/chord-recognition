import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import librosa.display
import mir_eval

def plotChromagram(ax,t_chroma,chroma,time_interval=None,vmin=0,vmax=1):
    i0 = 0
    i1 = -1
    if time_interval is not None:
        try:
            i0 = np.argwhere(t_chroma >= time_interval[0])[0][0]
            i1 = np.argwhere(t_chroma >= time_interval[1])[0][0]
        except IndexError:
            print(time_interval)
    img = librosa.display.specshow(chroma[:,i0:i1],x_coords=t_chroma[i0:i1],x_axis="time", y_axis='chroma', cmap="Reds", ax=ax,vmin=0, vmax=np.max(chroma[:,i0:i1]))
    return img

def plotChordAnnotations(ax,intervals,labels,time_interval=(0,10),format_label=False):
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

        rect = matplotlib.patches.Rectangle((t_start, 0),t_stop - t_start , 1.4, linewidth=1, edgecolor="k", facecolor=getColor(label))
        ax.add_patch(rect)
        if t_stop - t_start > 0.4:
            ax.text(t_start+ (t_stop - t_start)/2, 0.6, label,verticalalignment="center",horizontalalignment='center', fontsize=9, color='k')
    ax.set_ylim(0,2)
    ax.axis("off")
    ax.set_xlim(time_interval)

def plotResults(t_chroma,chroma,chroma_prefiltered,ref_intervals,ref_labels,est_intervals,est_labels):       
    fig,((ax0,ax01),(ax1,ax11),(ax2,ax21),(ax3,ax31)) = plt.subplots(4,2,height_ratios=(1,1,10,10),width_ratios=(20,.3),figsize=(9,7))
    plotChordAnnotations(ax0,ref_intervals,ref_labels,time_interval=(0,15))
    ax01.set_axis_off()
    plotChordAnnotations(ax1,est_intervals,est_labels,time_interval=(0,15))
    ax11.set_axis_off()
    img = plotChromagram(ax2,t_chroma,chroma,time_interval=(0,15))
    fig.colorbar(img,cax=ax21)
    img = plotChromagram(ax3,t_chroma,chroma_prefiltered,time_interval=(0,15))
    fig.colorbar(img,cax=ax31)
    return fig

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