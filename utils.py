import librosa
import librosa.display
import numpy as np
from matplotlib import rc

# Enable LaTeX support in matplotlib
# rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
# rc('text', usetex=True)

def formatChordLabel(chordlabel):
    # latex formatting of chordlabels
    try: 
        note,suffix = chordlabel.split(':')
    except ValueError:
        note = chordlabel
        suffix = ""
    try:
        chordtype,bassnote = suffix.split('/')
        if bassnote.startswith("b"):
            bassnote = bassnote[1]+"\\flat"
        bassnote = "/"+bassnote
    except ValueError:
        chordtype = suffix 
        bassnote = ""
    if suffix.startswith('maj7'):
        # Major seventh chord
        print(f"${note}^{{maj7}}{bassnote}$")
        return f"${note}^{{maj7}}{bassnote}$"
    elif suffix.startswith("maj"):
        # Major chord
        return f"${note}{bassnote}$"
    elif suffix.startswith('min7'):
        # Minor seventh chord    
        return f"${note}m^7{bassnote}$"
    elif suffix.startswith("min"):
        return f"${note}m{bassnote}$"
    elif suffix.startswith('7'):
        # Dominant seventh chord  
        return f"${note}^7{bassnote}$"
    elif suffix.startswith('aug'):
        # Augmented chord      
        return f"${note}^+$"
    elif suffix.startswith('dim'):
        # Diminished Chord
        return f"${note}^°$"
    else:
        # Major chord
        return note

def plotChroma(ax,chroma,time=(0,10),sr=44100,hop_length=4410):
    librosa.display.specshow(chroma.T,y_axis='chroma',cmap='viridis',
                                ax=ax,x_axis='time',sr=sr,hop_length=hop_length)
    ax.set_xlim(time)
    ax.set_xlabel('Time in s')

def plotCqt(ax,audiopath,time=(0,10)):
    y,_ = librosa.load(audiopath,sr=44100)
    cqt = librosa.cqt(y,hop_length=4410,sr=44100)
    librosa.display.specshow(librosa.amplitude_to_db(np.abs(cqt), ref=np.max),y_axis='cqt_note',cmap='viridis',
                                ax=ax,x_axis='time',sr=44100,hop_length=4410)
    ax.set_xlim((time))

def plotAudioWaveform(ax,y,time=(0,10)):
    t = np.linspace(0,(len(y)-1)/44100,len(y))
    mask = (t >= time[0]) & (t <= time[1])
    ax.plot(t[mask],y[mask]/np.max(y[mask]))
    ax.set_ylim([-1,1])
    ax.set_xlim(time[0],time[1])
    ax.set_xlabel('Time in s')
    ax.set_ylabel('Normalized amplitude')
    ax.grid('on')

def plotPredictionResult(ax,est_intervals,est_labels,time=(0,10)):
    #ax.text(time[0]-1.3,1.1,'Estimation:',fontsize=8,horizontalalignment='center',color='r')
    for i,label in enumerate(est_labels):
        if est_intervals[i,0] >= time[0] and est_intervals[i,0] < time[1]:
            ax.vlines(est_intervals[i,0],-1,1,'r',linestyles='dashed',linewidth=1)
            ax.text(est_intervals[i,0],1.1,label,fontsize=8,horizontalalignment='center',color='r')

def plotAnnotations(ax,ref_intervals,ref_labels,time=(0,10)):
    #ax.text(time[0]-1.3,1.2,'Label:',fontsize=8,horizontalalignment='center',color='k')
    for i,label in enumerate(ref_labels):
        if ref_intervals[i,0] >= time[0] and ref_intervals[i,0] < time[1]:
            ax.vlines(ref_intervals[i,0],-1,1,'grey',linestyles='dashed',linewidth=1)
            ax.text(ref_intervals[i,0],1.2,label,fontsize=8,horizontalalignment='center')
    ax.grid('off')