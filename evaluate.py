import dataloader
import mir_eval
import transcribe
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

def createChordLabelDict(alphabet="majmin"):
    pitch_class = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
    chord_labels = {}
    if alphabet == "majmin":
        # major chords
        for i,x in enumerate(pitch_class):
            chord_labels[f"{x}:maj"] = i     
        for i,x in enumerate(pitch_class):
            chord_labels[f"{x}:min"] = i+12 
        chord_labels["N"] = len(chord_labels)
    chord_labels["Others"] = len(chord_labels)
    return chord_labels

def compute_eval_measures(ref_evalmatrix, est_evalmatrix):
    """Compute evaluation measures
    Args:
        ref_evalmatrix (chords, Frames) : logical matrix of reference chords for each timestamp
        est_evalmatrix (chords, Frames) : logical matrix of estimated chords

    Returns:
        P (float): Precision
        R (float): Recall
        F (float): F-measure
        num_TP (int): Number of true positives
        num_FN (int): Number of false negatives
        num_FP (int): Number of false positives

        see
        https://meinardmueller.github.io/libfmp/build/html/index.html
    """
    TP = np.sum(np.logical_and(ref_evalmatrix, est_evalmatrix))
    FP = np.sum(est_evalmatrix > 0, axis=None) - TP
    FN = np.sum(ref_evalmatrix > 0, axis=None) - TP
    P = 0
    R = 0
    F = 0
    if TP > 0:
        P = round(TP / (TP + FP),2)
        R = round(TP / (TP + FN),2)
        F = round(2 * P * R / (P + R),2)
    return P, R, F, TP, FP, FN


def plotEvaluationMatrix(ref_evalmatrix,est_evalmatrix,title,transcriber):
    fig, ax = plt.subplots(figsize=(10,6))
    # create TP,FP,FN plot
    est_TP = np.sum(np.logical_and(ref_evalmatrix, est_evalmatrix))
    est_FP = est_evalmatrix - est_TP
    est_FN = ref_evalmatrix - est_TP
    I_vis = 3 * est_TP + 2 * est_FP + 1 * est_FN

    P, R, F, TP, FP, FN = compute_eval_measures(ref_evalmatrix,est_evalmatrix)
    
    eval_cmap = colors.ListedColormap([[1, 1, 1], [1, 0.3, 0.3], [1, 0.7, 0.7], [0, 0, 0]])
    eval_bounds = np.array([0, 1, 2, 3, 4])-0.5
    eval_norm = colors.BoundaryNorm(eval_bounds, 4)
    eval_ticks = [0, 1, 2, 3]
    im = ax.imshow(I_vis,  origin='lower', aspect='auto', cmap=eval_cmap, norm=eval_norm,
                      interpolation='nearest')
    plt.sca(ax)
    cbar = plt.colorbar(im, cmap=eval_cmap, norm=eval_norm, boundaries=eval_bounds, ticks=eval_ticks)
    cbar.ax.set_yticklabels(['TN', 'FP', 'FN', 'TP'])
    ax.set_xlabel("Time in Frames")
    chordlabels = createChordLabelDict()
    ax.set_yticks(list(chordlabels.values()))       
    ax.set_yticklabels(list(chordlabels.keys()))
    ax.set_ylabel("Chords")
    ax.set_title(f"{title} Precision: {P}, Recall: {R},F-measure: {F},TP: {TP}, FP:{FP}, FN:{FN}")
    plt.savefig(f"../transcription_results/{transcriber}_{title}.png")


def simplifyLabels(labels,alphabet="majmin"):
    enharmonic_notes = {"Db":"C#","Eb":"D#","Gb":"F#","Ab":"G#","Bb":"A#"}
    simplified_labels = []
    for label in labels:
        chord = mir_eval.chord.split(label, reduce_extended_chords=True)
        quality = chord[1]
        if chord[0] in enharmonic_notes:
            chord[0] = enharmonic_notes[chord[0]]
        if quality == "maj":
            simplified_labels.append(mir_eval.chord.join(chord[0],"maj"))
        elif quality == "min":
            simplified_labels.append(mir_eval.chord.join(chord[0],"min"))
        elif quality == "dim":
            simplified_labels.append(mir_eval.chord.join(chord[0],"dim"))
        elif chord[0] == "N":
            simplified_labels.append("N")
        else:
            # ignore dim and aug, sus for now..
            simplified_labels.append("Others")
    return simplified_labels

def createEvalMatrix(t_chroma,est_intervals,est_labels,ref_intervals,ref_labels):
    # adjust estimated to reference intervals and append "N" chord label if needed
    est_intervals, est_labels = mir_eval.util.adjust_intervals(
            est_intervals, est_labels, ref_intervals.min(),
            ref_intervals.max(), mir_eval.chord.NO_CHORD,
            mir_eval.chord.NO_CHORD)
    
    ref_labels = simplifyLabels(ref_labels,alphabet="majmin")

    chordlabels = createChordLabelDict("majmin")
    est_evalmatrix = np.zeros((len(chordlabels),t_chroma.shape[0]),dtype=int)
    ref_evalmatrix = np.zeros((len(chordlabels),t_chroma.shape[0]),dtype=int)
    for interval,label in zip(est_intervals,est_labels):
        try: 
            index_start = int(np.argwhere(t_chroma >= interval[0])[0])
            index_stop = int(np.argwhere(t_chroma >= interval[1])[0])
            est_evalmatrix[chordlabels[label],index_start:index_stop] = 1
        except KeyError:
            continue
    for interval,label in zip(ref_intervals,ref_labels):
        try: 
            index_start = int(np.argwhere(t_chroma >= interval[0])[0])
            index_stop = int(np.argwhere(t_chroma >= interval[1])[0])
            ref_evalmatrix[chordlabels[label],index_start:index_stop] = 1
        except KeyError:
            continue
    return ref_evalmatrix,est_evalmatrix

# oversegm and udersegm score

def evaluateTranscription(est_intervals,est_labels,ref_intervals,ref_labels):
    """evaluate Transcription score according to MIREX scheme"""
    est_intervals, est_labels = mir_eval.util.adjust_intervals(
            est_intervals, est_labels, ref_intervals.min(),
            ref_intervals.max(), mir_eval.chord.NO_CHORD,
            mir_eval.chord.NO_CHORD)
    (intervals,ref_labels,est_labels) = mir_eval.util.merge_labeled_intervals(
        ref_intervals, ref_labels, est_intervals, est_labels)
    durations = mir_eval.util.intervals_to_durations(intervals)
    comparisons = mir_eval.chord.thirds(ref_labels, est_labels)
    score = round(mir_eval.chord.weighted_accuracy(comparisons, durations),2)
    mean_seg_score = round(mir_eval.chord.seg(ref_intervals, est_intervals),2)
    return score,mean_seg_score
