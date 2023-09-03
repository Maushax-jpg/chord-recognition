import librosa
import features
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import mir_eval
import utils
import transcribe
import circularPitchSpace as cps
import itertools
import mir_eval
import madmom
import scipy.ndimage

plt.rcParams['text.usetex'] = True

def evaluateTranscription(est_intervals,est_labels,ref_intervals,ref_labels):
    """evaluate Transcription score according to MIREX scheme"""
    est_intervals, est_labels = mir_eval.util.adjust_intervals(
            est_intervals, est_labels, ref_intervals.min(),
            ref_intervals.max(), mir_eval.chord.NO_CHORD,
            mir_eval.chord.NO_CHORD)
    (intervals, ref_labels, est_labels) = mir_eval.util.merge_labeled_intervals(
        ref_intervals, ref_labels, est_intervals, est_labels)
    durations = mir_eval.util.intervals_to_durations(intervals)
    comparisons = mir_eval.chord.thirds(ref_labels, est_labels)
    score = round(mir_eval.chord.weighted_accuracy(comparisons, durations),2)
    mean_seg_score = round(mir_eval.chord.seg(ref_intervals, est_intervals),2)
    return score,mean_seg_score

def labelChord(combination):
    if not combination:
        return "N"
    root = cps.pitch_classes[combination[0]].name
    quality=""
    # calculate intervals, add offset if needed
    intervals = [y-x if y>x else y-x+12 for x,y in itertools.pairwise(combination)]
    
    if len(intervals) == 2: # TRIAD
        if intervals ==  [4,3]:
            quality = "maj"
        elif intervals ==  [3,4]:
            quality = "min"
        elif intervals ==  [3,3]:
            quality = "dim"
        elif intervals == [4,4]:
            quality = "aug"
    elif len(intervals) == 3: # TETRAD
        if intervals ==  [4,3,3]:
            quality = "7"
        elif intervals ==  [4,3,4]:
            quality = "maj7"
        elif intervals ==  [3,4,3]:
            quality = "min7"
        elif intervals == [3,4,4]:
            quality = "minmaj7"
        elif intervals == [3,3,3]:
            quality = "dim7"
        elif intervals == [3,3,4]:
            quality = "hdim7"
    elif len(intervals) > 3: # five-note chord
        quality="9"
        # add more if needed!
    return mir_eval.chord.join(root, quality)

def stackThirds(notes):
    for note_combination in itertools.permutations(notes,len(notes)):
        valid_chord = True
        for interval in itertools.pairwise(note_combination):
            diff = interval[1] - interval[0]
            if(interval[1] - interval[0]) < 0:
                diff = interval[1] - interval[0] + 12 
            if diff < 3 or diff > 4:
                valid_chord = False
                break
        if valid_chord:
            break
    if valid_chord:
        return note_combination
    else:
        return None

def getBitmap(chordlabel):
    root,quality,_,_ = mir_eval.chord.split(chordlabel)
    if root == "N":
        return np.ones((12,),dtype=int)
    root =  mir_eval.chord.pitch_class_to_semitone(root)
    bitmap = mir_eval.chord.quality_to_bitmap(quality)
    return mir_eval.chord.rotate_bitmap_to_root(bitmap,root)

def processPrediction(time_vector,labels):
    est_intervals = []
    est_labels = []
    t_start = 0
    for i,(label_t0,label_t1) in enumerate(itertools.pairwise(labels)):
        if label_t0 != label_t1: # potential chord change
            est_labels.append(label_t0)
            try:
                est_intervals.append([t_start,time_vector[i+1]])
                t_start = time_vector[i+1]
            except Exception as e:
                print(e)
    # add last chord interval
    if label_t0 == label_t1:
        est_labels.append(label_t0)
        est_intervals.append([t_start,time_vector[-1]])
    return est_intervals, est_labels

def postprocessing(time_vector,unfiltered_labels,minimum_duration=0.4):
    intervals = []
    labels = []
    t_start = 0
    current_label = "N"
    for i,label in enumerate(unfiltered_labels[:-1]):
        if label == current_label:
            continue
        # check duration of chord change (excluding first chord change!)
        if time_vector[i] - t_start < minimum_duration:
            # interval too short, extend previous interval  and discard chord change
            if intervals:
                intervals[-1][1] = time_vector[i]
        else:
            labels.append(current_label)
            intervals.append([t_start,time_vector[i]])
        # initialize new interval
        current_label = label
        t_start = time_vector[i]
    labels.append(label)
    intervals.append([t_start,time_vector[i+1]])
    intervals = np.array(intervals)
    return intervals,labels

start = 0
stop = 15
path = "/home/max/ET-TI/Masterarbeit/mirdata/beatles/"
title = "12_-_Let_It_Be/06_-_Let_It_Be"
# title = "06_-_Rubber_Soul/11_-_In_My_Life"
# title = "10CD1_-_The_Beatles/CD1_-_17_-_Julia"

y,sr = librosa.load(path+"/audio/"+title+".wav",offset=start,duration=stop-start)

beats = None
if beats:
    beats = features.RNN_beats(path+"/audio/"+title+".wav")
    beats = [x for x in beats if x >= start and x <= stop]
    print(beats) 

# load labels and adjust time span
ref_intervals,ref_labels = mir_eval.io.load_labeled_intervals(path+"/annotations/chordlab/The Beatles/"+title+".lab",' ','#')
i_start = 0
i_stop = 0
start_flag = True
for i in range(ref_intervals.shape[0]):
    if start_flag and ref_intervals[i][0] >= start:
        i_start = i
        start_flag = False
    if ref_intervals[i][0] >= stop:
        i_stop = i
        break
ref_intervals = ref_intervals[i_start:i_stop]
ref_labels = ref_labels[i_start:i_stop]

chroma = features.crpChroma(y)

# filter chromagram
# chroma_filter = np.minimum(chroma,
#                                     librosa.decompose.nn_filter(chroma,
#                                     axis=0,
          
#                                     aggregate=np.median,
#                                     metric='cosine'))
# # apply horizontal median filter
# chroma = scipy.ndimage.median_filter(chroma_filter, size=(5, 1))
# chroma = chroma / (np.expand_dims(np.sum(chroma,axis=1), axis=1)+np.finfo(float).eps)

#chroma = features.librosaChroma(y)

time_vector = np.linspace(start, stop, chroma.shape[0], endpoint=False) 


# key analysis with circular pitch space
key = cps.estimateKey(chroma,alpha=0.95,threshold=0.8,angle_weight=0.5)
labels = []
for t in range(time_vector.shape[0]):
    # split relevant chromabins into key related notes and non_harmonic_notes
    diatonic_notes, non_harmonic_notes = cps.extractNotes(chroma[t,:],key[t])
    # find a combination of thirds that forms a diatonic chord with all possible diatonic notes
    combination = stackThirds(diatonic_notes)
    # if there is still no combination, try to include non_harmonic_notes
    if (not combination or len(combination) < 3) and non_harmonic_notes:
        combination = stackThirds(diatonic_notes+non_harmonic_notes)
    # label the chord by analysing the intervals
    labels.append(labelChord(combination))

# calculate intervals
est_intervals, est_labels = postprocessing(time_vector,labels)

maj_min_score,segmentation = evaluateTranscription(est_intervals,est_labels,ref_intervals,ref_labels)
print(f"MajMin: {round(100*maj_min_score,1)}%")
# display results
fig = plt.figure(figsize=(10, 5))
gs = matplotlib.gridspec.GridSpec(4, 2,width_ratios=(40,1),height_ratios=(1,1,1,6), hspace=0.5,wspace=0.05)
utils.plotChordAnnotations(fig.add_subplot(gs[0,0]),(ref_intervals,ref_labels),(start,stop))
utils.plotChordAnnotations(fig.add_subplot(gs[1,0]),(est_intervals,est_labels),(start,stop))
#utils.plotChordAnnotations(fig.add_subplot(gs[2,0]),(intervals,labels),(start,stop))
ax00 = fig.add_subplot(gs[3,0])
ax01 = fig.add_subplot(gs[3,1])
img = utils.plotChromagram(ax00,time_vector,chroma,beats)
fig.colorbar(img,cax=ax01)
plt.show()

