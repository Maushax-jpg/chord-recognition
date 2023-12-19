import librosa.display
import mir_eval
import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
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

def loadAudiofile(filepath,fs=22050):
    """load audio signal"""
    try:
        y,_ = librosa.load(filepath,mono=True,sr=fs)
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
    for i in range(12):
        templates[i,chord_index] = -1/12
    chord_labels.append("N")
    return templates,chord_labels

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
    img = librosa.display.specshow(cqt[:,i0:i1],x_coords=t_chroma[i0:i1],x_axis="time", y_axis='cqt', cmap="viridis", ax=ax,vmin=0, vmax=np.max(cqt[:,i0:i1]))
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