from argparse import ArgumentParser
import os
import h5py
import numpy as np
import mir_eval
import dataloader
import utils
from tqdm import tqdm
import features
from features import computeCorrelation
from scipy.stats import pearsonr

def computeCorrelation(chromadata,templates):
    """compute correlation of chromavectors (12,A) with the specified templates (12,B)

    returns a correlation matrix of shape (B,A)
    """
    correlation = np.zeros((templates.shape[1],chromadata.shape[1]),dtype=float)
    # using scipy pearsons correlation coefficient
    for i in range(templates.shape[1]):
        for t in range(chromadata.shape[1]):
            correlation[i,t] = pearsonr(templates[:,i],chromadata[:,t]).statistic

    # replace NaN with zeros
    correlation = np.nan_to_num(correlation)
    np.clip(correlation,out=correlation,a_min=0,a_max=1)  
    return correlation

if __name__ == "__main__":
    parser = ArgumentParser(prog='Chromadata extractor', description='creates labeled chromadata from existing chromagrams')

    parser.add_argument('chroma',choices=["crp","dcp"])
    parser.add_argument('filter_type',choices=["max","thresh"])
    args = parser.parse_args()

    # create a default path 
    script_directory = os.path.dirname(os.path.abspath(__file__))
    datasetpath = os.path.join(script_directory, "mirdata")
    outputpath = os.path.join(script_directory, "models","chromadata",f"chromadata_{args.chroma}_{args.filter_type}.hdf5")

    print(f"create templates for triads_tetrads alphabet")
    templates, labels = utils.createChordTemplates("triads_tetrads")

    print("Load chromadata")
    chords = {}
    chroma_params = {}
           
    # iterate over all datasets
    for dset_name in ["beatles","rwc_pop","rw","queen"]:
        dataset = dataloader.Dataloader(dset_name,datasetpath,"none") 
        for split in range(1,9):
            for track_id in tqdm(dataset.getExperimentSplits(split),desc=f"{dset_name}_{split}"):
                audiopath,annotationpath = dataset[track_id]

                if args.chroma == "crp":
                    if chroma_params: # only access once
                        # if not specified the default values were used
                        chroma_params["nCRP"] = 33
                        chroma_params["eta"] = 100
                        chroma_params["hop_length"] = 2048
                    y = utils.loadAudiofile(audiopath)
                    chroma = features.crpChroma(y)
                    t_chroma = utils.timeVector(chroma.shape[1],hop_length=2048)
                else:
                    chroma = features.deepChroma(audiopath,split)
                    t_chroma = utils.timeVector(chroma.shape[1],hop_length=2205) # 100ms 

                ref_intervals,ref_labels = utils.loadChordAnnotations(annotationpath)

                # iterate over all ground_truth labels
                for interval,label in zip(ref_intervals,ref_labels):
                    # compute time indices to access the chromagram at the correct index
                
                    i0,i1 = utils.getTimeIndices(t_chroma,interval)
                    chromadata = np.array(chroma[:,i0:i1])
                    root,qual,scale_deg,bass = mir_eval.chord.split(label,True)

                    if root == "N":
                        chordlabel = "N"
                    else: # shift chroma to root note C 
                        pc_index = mir_eval.chord.pitch_class_to_semitone(root)
                        chromadata = np.roll(chromadata,-pc_index,axis=0)
                        chordlabel = mir_eval.chord.join("C",qual)

                    # check if chordlabel exists in specified vocabulary
                    try:
                        chord_index = labels.index(chordlabel)
                    except ValueError:
                        # chordlabel not in list, skip to next label 
                        continue
                    # filter out invalid chromadata (because the labels are hand annotated, errors may occur)
                    # only keep datapoints where the correlation with the annotated label is high
                    # only keep best fitting points or discard highly unlikely chords
                    corr = computeCorrelation(chromadata,templates)
                    # highest correlation
                    if args.filter_type == "max":
                        max_corr_index = np.argmax(corr, axis=0)
                        mask = [True if index == labels.index(chordlabel) else False for index in max_corr_index]
                    # threshold
                    elif args.filter_type == "thresh":
                        mask = corr[labels.index(chordlabel), :] > 0.6

                    # append the current chromadata to the data entries in the dictionary 
                    data = chords.get(chordlabel)
                    if data is None:
                        chords[chordlabel] = chromadata[:, mask]
                    else:
                        chords[chordlabel] = np.concatenate((data,chromadata[:, mask]),axis=1)

    print("write labeled chromadata to file")
    with  h5py.File(outputpath,"w") as file:
        for key,value in chroma_params.items():
            file.attrs.create(key,value)
        for key,value in chords.items():
            subgrp = file.create_group(key)
            subgrp.create_dataset(key,data=value)