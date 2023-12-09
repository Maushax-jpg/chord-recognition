import librosa.display
import mir_eval
import itertools
import numpy as np
from collections import namedtuple
import audioread.ffdec  # Use ffmpeg decoder

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
    """load audio signal with ffmpeg decoder"""
    try:
        aro = audioread.ffdec.FFmpegAudioFile(filepath)
        y,_ = librosa.load(aro,mono=True,sr=fs)
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