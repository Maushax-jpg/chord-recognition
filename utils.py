import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def buffer(X: np.ndarray, n: int, p: int,zeros : int,window : np.ndarray) -> np.ndarray:
    # buffers data vector X into length n column vectors with overlap p
    # and zero padding on both ends of the signal
    # excess data at the end of X is discarded
    L = len(X)  # length of data to be buffered
    m = int(np.floor((L - n) / (n - p)) + 1)  # number of sample vectors (no padding)
    data = np.zeros((n+zeros, m))  # initialize data matrix
    startIndex = int(zeros/2)
    if n <= 0 or p < 0 or p >= n:
        raise ValueError("Invalid buffer parameters")
    for i, column in zip(range(0, L - n + 1, n - p), range(0, m)):
        data[startIndex : startIndex + n, column] = X[i : i + n] * window
    return data


def formatChordLabel(chordlabel):
    # latex formatting of chordlabels
    try: 
        note,suffix = chordlabel.split(':')
    except ValueError:
        note = chordlabel
        suffix = ""
    if note[-1] == '#':
        note = note[:-1]+"\\#"
    try:
        chordtype,bassnote = suffix.split('/')
        if bassnote.startswith("b"):
            bassnote = bassnote[1]+"\\flat"
        bassnote = "/"+bassnote
    except ValueError:
        chordtype = suffix 
        bassnote = ""
    if chordtype.startswith('maj7'):
        # Major seventh chord
        return f"${note}^{{maj7}}{bassnote}$"
    elif chordtype.startswith("maj"):
        # Major chord
        return f"${note}{bassnote}$"
    elif chordtype.startswith('min7'):
        # Minor seventh chord    
        return f"${note}m^7{bassnote}$"
    elif chordtype.startswith("min"):
        return f"${note}m{bassnote}$"
    elif chordtype.startswith('7'):
        # Dominant seventh chord  
        return f"${note}^7{bassnote}$"
    elif chordtype.startswith('aug'):
        # Augmented chord      
        return f"${note}^+$"
    elif chordtype.startswith('dim'):
        # Diminished Chord
        return f"${note}^°$"
    else:
        # Major chord
        return f"${note}$"

def plotChromagram(ax,t,chroma, beats=None):
    librosa.display.specshow(chroma.T,x_coords=t.T,x_axis='time', y_axis='chroma', cmap="Reds", ax=ax, vmin=0, vmax=1)
    if beats is not None:
        ax.vlines(beats,-0.5,11.5,'k',linestyles='dashed',alpha=0.2)

def plotChordAnnotations(ax,target,estimation=None,time_interval=(0,10)):
    ref_intervals, ref_labels = target
    est_intervals,est_labels = target # estimation not implemented yet
    colors = ["red", "green", "blue", "yellow", "orange", "purple", "pink", "cyan", "brown", "magenta", "teal", "gray"]
    for i,label in enumerate(ref_labels):
        # skip labels that do not overlap time interval 
        if ref_intervals[i,1] < time_interval[0] or ref_intervals[i,0] > time_interval[1]:
            continue
        # set start position of rectangular patch
        t_start = max(ref_intervals[i,0],time_interval[0])
        t_stop = min(ref_intervals[i,1],time_interval[1])

        rect = patches.Rectangle((t_start, 0),t_stop - t_start , 1.2, linewidth=1, edgecolor="k", facecolor=colors[i%len(colors)])
        ax.add_patch(rect)
        ax.text(t_start+ (t_stop - t_start)/2, 0.6, formatChordLabel(label),verticalalignment="center",horizontalalignment='center', fontsize=10, color='k')
    ax.set_xlim(time_interval)
    ax.set_ylim(0,2)
    ax.axis("off")
    
def plotCqt(ax,signal):
    cqt = librosa.cqt(signal,hop_length=4410,sr=44100)
    librosa.display.specshow(librosa.amplitude_to_db(np.abs(cqt), ref=np.max),y_axis='cqt_note',cmap='viridis',
                                ax=ax,x_axis='time',sr=44100,hop_length=4410)
                                
def plotAudioWaveform(ax,y,time=(0,10)):
    t = np.linspace(0,(len(y)-1)/44100,len(y))
    mask = (t >= time[0]) & (t <= time[1])
    ax.plot(t[mask],y[mask]/np.max(y[mask]))
    ax.set_ylim([-1,1])
    ax.set_xlim(time[0],time[1])
    ax.set_xlabel('Time in s')
    ax.set_ylabel('Normalized amplitude')
    ax.grid('on')
