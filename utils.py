import matplotlib.pyplot as plt
import librosa.display
import numpy as np
from matplotlib import rc

# custom
import dataloader

# Enable LaTeX support in matplotlib
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

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

def plotAudioWaveform(ax,audiopath,interval):
    start_time,end_time = interval
    y,sr = librosa.load(audiopath,sr=44100)
    t = np.linspace(0,(len(y)-1)/sr,len(y))
    mask = (t >= start_time) & (t <= end_time)
    ax.plot(t[mask],y[mask]/np.max(y[mask]))
    ax.set_ylim([-1,1])
    ax.set_xlim(start_time,end_time)
    ax.set_xlabel('Time in s')
    ax.set_ylabel('Normalized amplitude')
    ax.grid('on')

def plotPredictionResult(ax,predictions_idx,predictions,interval):
    start_time,end_time = interval
    ax.text(start_time-1.3,1.1,'Estimation:',horizontalalignment='center',color='r')
    for i,index in enumerate(predictions_idx):
        t = float(index)*0.1  # conversion index to time 
        if (t >= start_time) & (t <= end_time):
            ax.vlines(t,-1,1,'r',linestyles='dashed',linewidth=1)
            ax.text(t,1.1,formatChordLabel(predictions[i]),horizontalalignment='center',color='r')

def plotAnnotations(ax,annotationspath,interval,chords='majmin',dataset='beatles'):
    start_time,end_time = interval
    if not dataset=='beatles':
        raise NotImplementedError
    # load annotations
    labels_df= dataloader.getBeatlesAnnotations(annotationspath)
    # if chords == 'majmin':
    #     labels_df = dataloader.simplifyAnnotations(labels_df,'majmin')
    df = labels_df[(labels_df['tstart'] >= start_time) & (labels_df['tstart'] <= end_time)]
    ax.text(start_time-1.3,1.2,'Label:',horizontalalignment='center',color='k')
    for t,label in zip(df['tstart'],df['label']):
        ax.vlines(t,-1,1,'grey',linestyles='dashed',linewidth=1)
        ax.text(t,1.2,formatChordLabel(label),horizontalalignment='center')
