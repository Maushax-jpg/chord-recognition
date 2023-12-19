import dataloader
import numpy as np
import argparse
import os
import h5py
import utils
import mir_eval
import itertools
import matplotlib.pyplot as plt

def bigramAnalysis(datasets,alphabet="majmin",split=1,dataset_path=None):
    def getIndex(chordlabel,alphabet="majmin"):
        root,quality,_,_ = mir_eval.chord.split(chordlabel)
        pitch_class_offset = 0
        if alphabet == "majmin":
            if quality in ["dim","hdmin7","min7","minmaj7","min6","min9","min11","min13"]:
                pitch_class_offset = 12
        else:
            raise NotImplementedError("sevenths alphabet not implemented yet!")
        pc_index = mir_eval.chord.pitch_class_to_semitone(root)
        return pc_index,pitch_class_offset

    hop_time = 2048/22050
    if alphabet == "majmin":
        A = np.zeros((25,25),dtype=int)
    if dataset_path is None:
        # use default path
        script_directory = os.path.dirname(os.path.abspath(__file__))
        dataset_path = os.path.join(script_directory, "mirdata")
    for dataset_name in datasets:
        dset = dataloader.Dataloader(dataset_name,dataset_path,None)
        for i in range(1,9):
            if i == split: 
                continue
            for track_id in dset.getExperimentSplits(i):
                audiopath,annotationpath = dset[track_id]
                ref_intervals,ref_labels = utils.loadChordAnnotations(annotationpath)
                for intervals,labels in zip(itertools.pairwise(ref_intervals),itertools.pairwise(ref_labels)):
                    if labels[0] == "N" or labels[1] == "N":
                        continue
                    # compute the indices for the state transition matrix
                    pc_index_0, pc_offset_0 = getIndex(labels[0])
                    pc_index_1, pc_offset_1 = getIndex(labels[1])
                    # calculate the number of self transitions 
                    A[pc_offset_0,pc_offset_0] += (intervals[0][1]-intervals[0][0]) // hop_time
                    # add chord transition
                    A[pc_offset_0,pc_index_1 - pc_index_0 + pc_offset_1] += 1
    
    A[-1,-1] = A[12,12] = A[0,0]
    A[-1,:-1] = A[:-1,-1] = np.min(A[0,:])
    A = A / np.sum(A)
    for i in range(1,12):
        A[i,:-1] = np.roll(A[0,:-1],i)
    for i in range(1,12):
        A[i+12,:-1] = np.roll(A[12,:-1],i)
    return A


if __name__ == "__main__":
    print("Compute Chord statistics from .hdf5 result file")
    parser = argparse.ArgumentParser(prog='Automatic chord recognition', description='Transcribe audio signal')

    # create a default path 
    script_directory = os.path.dirname(os.path.abspath(__file__))
    default_path = os.path.join(script_directory, "mirdata")

    parser.add_argument('--filepath', default=None, help="specify path to .hdf5 file from which to compute the chord statistics")
    parser.add_argument('--outputpath',default=None, help="specify output path, where to write the output  .hdf5 file")
    parser.add_argument("--vocabulary",default="majmin",help="specify the chord alphabet: majmin, triads, triads_extended, 7, 7_triads or 7_triads_extended")
    args = parser.parse_args()

    # list of tracks that should be excluded due to issues with the annotations / or lack of harmonic content
    outliers = ['03_-_Anna_(Go_To_Him)', # tuning issues
                '10_-_Lovely_Rita', # tuning issues
                'CD1_-_05_-_Wild_Honey_Pie', # little harmonic content
                'CD1_-_06_-_The_Continuing_Story_of_Bungalow_Bill', # incoherent audiofile
                'CD2_-_12_-_Revolution_9',  # little harmonic content
                '02 Another One Bites The Dust', # little harmonic content
                '16 We Will Rock You', # little harmonic content
                "Stalker's Day Off (I've Been Hanging Around)", # faulty audio / issues with annotations
                'Stand Your Ground'  # faulty audio / issues with annotations
    ]

    # overwrite args for testing
    filepath = "/home/max/ET-TI/Masterarbeit/chord-recognition/results/results_recurrencePlot.hdf5"
    outputpath = "/home/max/ET-TI/Masterarbeit/chord-recognition/models/chromadata_root_invariant_prefiltered_split_1.hdf5"
    alphabet = "majmin"
    split = 1


    # The annotations are to granular for the chromagram we can compute, therefore one can ignore additonal notes
    ignore_additional_notes = True
    # compute root invariant vectors
    shift_to_c =  True
    # compute a mean chromavector
    compute_median = True
    # use prefilter
    prefitered = True

    if alphabet == "majmin":
        qualities = ["N","maj","min"]
    elif alphabet == "triads":
        qualities = ["N","maj","min","dim","aug"]
    elif alphabet == "triads_extended":
        qualities = ["N","maj","min","dim","aug","sus2","sus4"]
    elif alphabet == "7":
        qualities = ["N","maj","min","maj7","7","min7"]
    elif alphabet == "7_triads":
        qualities = ["N","maj","min","dim","aug","maj7","7","min7"]

    with  h5py.File(filepath,"r") as file:
        datasets = file.attrs.get("dataset")

        if datasets is None:
            raise KeyError("Corrupt result file! no datasets are specified in the header!")
        
        if not isinstance(datasets,list): # convert to list if necessary
            datasets = list(datasets)

        chords = {}
        chroma_params = {}
        prefilter = file.attrs.get("prefilter")
        # iterate over all datasets
        for grp_name in datasets:
            dset = dataloader.Dataloader(grp_name,default_path,None)
            
            
            # track_list = dset.getExperimentSplits(split) # list of track_ids that are in this fold/split

            # iterate over all track_ids
            for subgrp_name in file[f"{grp_name}/"]:
                subgrp = file[f"{grp_name}/{subgrp_name}"]
                
                # if subgrp_name in track_list: # exclude the track
                #     continue

                # load chroma and ground truth
                if prefitered:
                    chroma = subgrp.get("chroma_prefiltered")
                else:
                    chroma = subgrp.get("chroma")

                if chroma_params: # only access once
                    # if not specified the default values were used
                    chroma_params["nCRP"] = subgrp.get("nCRP",33) 
                    chroma_params["eta"] = subgrp.get("eta",100)

                ref_intervals = subgrp.get("ref_intervals")
                ref_labels = subgrp.get("ref_labels")
                hop_length = chroma.attrs.get("hop_length")
                if hop_length is None:
                    raise ValueError("No chromagram hopsize found!")
                t_chroma = utils.timeVector(chroma.shape[1],hop_length=hop_length)

                # iterate over all ground_truth labels
                for interval,label in zip(ref_intervals,ref_labels):
                    # compute time indices to access the chromagram at the correct index
                
                    i0,i1 = utils.getTimeIndices(t_chroma,interval)
                    if compute_median:
                        # copy array and reshape it to size (12,1)
                        temp_chroma = np.array(np.median(chroma[:,i0:i1],axis=1)).reshape(-1,1)
                    else:
                        temp_chroma = np.array(chroma[:,i0:i1])
                    root,qual,scale_deg,bass = mir_eval.chord.split(label.decode("utf-8"))

                    if root == "N":
                        temp_label = "N"
                    else: # shift chroma to root note C 
                        pc_index = mir_eval.chord.pitch_class_to_semitone(root)
                        if shift_to_c:
                            temp_chroma = np.roll(temp_chroma,-pc_index,axis=0)
                            root="C"
                        if ignore_additional_notes:
                            temp_label = f"{root}-{qual}-set()-1"  # root is bass note and no additional notes are used 
                        else:
                            temp_label = f"{root}-{qual}-{scale_deg}-{bass}"  # create a label that can be decoded easily

                    # append chromasequences to data if possible
                    data = chords.get(temp_label)
                    if data is None:
                        chords[temp_label] = temp_chroma
                    else:
                        chords[temp_label] = np.concatenate((data,temp_chroma),axis=1)
       
    with  h5py.File(outputpath,"w") as file:
        file.attrs.create("split",split)
        file.attrs.create("prefilter",prefilter)
        for key,value in chroma_params.items():
            file.attrs.create(key,value)

        for key,value in chords.items():
            grp = file.create_group(key)
            grp.create_dataset(key,data=value)