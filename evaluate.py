import dataloader
import HMM
import mir_eval
import madmom
import numpy as np

if __name__ == '__main__':
    dataset_path = "/home/max/ET-TI/Masterarbeit/datasets/beatles/"
    dataset = dataloader.BeatlesDataset(dataset_path,'madmom',beat_align=True)
    model_path = "/home/max/ET-TI/Masterarbeit/models/hmm_model_madmom.pkl"   

    model = HMM.load_model(model_path)

    # Chord Recognition Processor
    crp = madmom.features.chords.DeepChromaChordRecognitionProcessor()
    decoder = 'hmm'

    songs = [100,101,102,103]  
    
    for index in songs:
        t_chroma,chroma,ref_intervals,ref_labels = dataset[index]
        if decoder == 'hmm':
            # HMM prediction
            est_intervals,est_labels = model.predict(t_chroma,chroma)
        else:
            prediction = crp(chroma)
            # prediction ->  array([(tstart , stop, label), .. ,(tstart , stop, label)])
            est_intervals = np.array([[x[0],x[1]] for x in prediction])
            est_labels = [x[2] for x in prediction]

        est_intervals, est_labels = mir_eval.util.adjust_intervals(
            est_intervals, est_labels, ref_intervals.min(),
            ref_intervals.max(), mir_eval.chord.NO_CHORD,
            mir_eval.chord.NO_CHORD)
        (intervals,ref_labels,est_labels) = mir_eval.util.merge_labeled_intervals(
            ref_intervals, ref_labels, est_intervals, est_labels)
        durations = mir_eval.util.intervals_to_durations(intervals)
        comparisons = mir_eval.chord.thirds(ref_labels, est_labels)
        score = mir_eval.chord.weighted_accuracy(comparisons, durations)
        print(f"{dataset.getTitle(index)}: {round(score*100,2)}%")