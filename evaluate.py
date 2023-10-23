import mir_eval

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