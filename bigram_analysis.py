import mir_eval
import numpy as np
import os
import itertools
from argparse import ArgumentParser
import dataloader
import utils

DATASETS = ["beatles", "rwc_pop","rw","queen"]

HOP_TIME = 2048/22050  # hopsize/fs

ENHARMONIC_NOTES = {"C#":"Db",
                    "D#":"Eb",
                    "E#":"F",
                    "F#":"Gb",
                    "G#":"Ab",
                    "A#":"Bb",
                    "B#":"C"}

QUALITY_REDUCTION = {"min6":"min",
                     "maj6":"maj",
                     "7":"maj",
                     "sus4":"maj",
                     "sus2":"min",
                     "maj7":"maj",
                     "min7":"min",
                     "dim":"min",
                     "aug":"maj",
                     "hdim7":"min",
                     "dim7":"min",
                     "":"maj",
                     "5":"maj"}

if __name__ == "__main__":
    parser = ArgumentParser(prog='Bigram analysis', description='creates a state-transition matrix')

    parser.add_argument('alphabet',default="majmin",choices=["majmin","sevenths"])
    args = parser.parse_args()

    if args.alphabet == "sevenths":
        del QUALITY_REDUCTION["maj7"]
        del QUALITY_REDUCTION["7"]
        del QUALITY_REDUCTION["min7"]

    # create a default path 
    script_directory = os.path.dirname(os.path.abspath(__file__))
    default_path = os.path.join(script_directory, "mirdata")

    _,labels = utils.createChordTemplates(args.alphabet)
    A = np.ones((len(labels),len(labels)),dtype=int)
    for dataset_name in DATASETS:
        dset = dataloader.Dataloader(dataset_name,default_path,"none")
        for track_id in dset.getTrackList():
            audiopath,annotationpath = dset[track_id]
            ref_intervals,ref_labels = utils.loadChordAnnotations(annotationpath)
            for (interval_a,interval_b), (label_a, label_b) in zip(itertools.pairwise(ref_intervals),itertools.pairwise(ref_labels)):
                if label_a == "N" or label_b == "N": # ignore no chord changes
                    continue

                # reduce extended chordlabels and ignore additional notes
                root_a,qual_a,_,_ = mir_eval.chord.split(label_a,True)
                root_b,qual_b,_,_ = mir_eval.chord.split(label_b,True)
                qual_a = QUALITY_REDUCTION.get(qual_a,qual_a)
                qual_b = QUALITY_REDUCTION.get(qual_b,qual_b)

                pc_root_a = mir_eval.chord.pitch_class_to_semitone(root_a)
                pc_root_b =  mir_eval.chord.pitch_class_to_semitone(root_b)

                # iterate all 12 keys
                for pc_index in range(12):
                    shift = pc_root_a - utils.pitch_classes[pc_index].pitch_class_index
                    # set root of label_a to "C", shift root of label_b accordingly
                    root_a = utils.pitch_classes[pc_index].name
                    root_b = utils.pitch_classes[(pc_root_b - shift) % 12].name
                    
                    # perform enharmonic substitution if possible
                    root_b = ENHARMONIC_NOTES.get(root_b,root_b)
                    chordlabel_a = mir_eval.chord.join(root_a, qual_a)
                    chordlabel_b = mir_eval.chord.join(root_b, qual_b)

                    index_a = labels.index(chordlabel_a)
                    index_b = labels.index(chordlabel_b)
                    # calculate the number of self transitions for label_a
                    A[index_a, index_a] += (interval_a[1] -interval_a[0]) // HOP_TIME
                    # add chord transition
                    A[index_a, index_b] += 1
                    
    # set some values for No chord transition
    A[-1,-1] = A[0,0]
    A[-1,:-1] = A[:-1,-1] = np.min(A[0,:])

    # normalize the matrix to 1 
    state_transition_matrix = np.zeros_like(A,dtype=float)
    for i in range(state_transition_matrix.shape[0]):
        state_transition_matrix[i,:] = A[i,:] / np.sum(A[i,:])

    # save to file
    outputpath = os.path.join(script_directory, "models","state_transitions",f"A_{args.alphabet}.npy")
    np.save(outputpath, state_transition_matrix)