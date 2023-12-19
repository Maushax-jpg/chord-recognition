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
        rect = matplotlib.patches.Rectangle((t_start, y_0),t_stop - t_start , 1.4, linewidth=1, edgecolor="k", facecolor=getColor(label))
        ax.add_patch(rect)
        if t_stop - t_start > 0.5:
            ax.text(t_start+ (t_stop - t_start)/2, y_0 + 0.6, label,verticalalignment="center",horizontalalignment='center', fontsize=9, color='k')
    ax.set_xlim(time_interval)

def plotResults(chromadata,ground_truth,majmin_estimation,sevenths_estimation,time_interval=(0,20)):   
    
    t_chroma,chroma,chroma_prefiltered = chromadata  
    ref_intervals,ref_labels = ground_truth
    est_majmin_intervals,est_majmin_labels = majmin_estimation
    est_sevenths_estimation_intervals,est_sevenths_estimation_labels = sevenths_estimation
    fig,((ax1,ax11),(ax2,ax21),(ax3,ax31)) = plt.subplots(3,2,height_ratios=(5,5,5),width_ratios=(20,.3),figsize=(9,6),sharex=False)
    # plot annotations
    plotChordAnnotations(ax1,ref_intervals,ref_labels,time_interval=time_interval,y_0=6)
    ax1.text(time_interval[0],7.7,"Ground truth annotations")
    plotChordAnnotations(ax1,est_majmin_intervals,est_majmin_labels,time_interval=time_interval,y_0=3)
    ax1.text(time_interval[0],4.7,"Estimated with majmin alphabet")
    plotChordAnnotations(ax1,est_sevenths_estimation_intervals,est_sevenths_estimation_labels,time_interval=time_interval,y_0=0)
    ax1.text(time_interval[0],1.7,"Estimated with sevenths alphabet")
    ax1.set_ylim(0,8)
    ax1.set_yticks([])
    # Hide the y-axis line
    ax1.spines['left'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.axis("on")
    ax11.set_axis_off()

    img = plotChromagram(ax2,t_chroma,chroma_prefiltered,time_interval=time_interval)
    fig.colorbar(img,cax=ax21)
    img = plotChromagram(ax3,t_chroma,chroma,time_interval=time_interval)
    fig.colorbar(img,cax=ax31)

    # create a label for the estimates
    xticks = np.linspace(time_interval[0],time_interval[1],21)
    xticklabels = [xticks[i] if i % 5 == 0 else "" for i in range(21)]
    for ax in [ax1,ax2,ax3]:
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        ax.set_xlabel("")

    ax3.set_xlabel("Time in s")
    fig.tight_layout(h_pad=0.1,w_pad=0.1,pad=0.3)
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