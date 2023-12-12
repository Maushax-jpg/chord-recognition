import os
import argparse
import dataloader
import utils
import features
import numpy as np
import plots
import h5py
from tqdm import tqdm
from datetime import datetime
import json

def parse_arguments():
    """extract command line arguments in ordert to setup chord recognition pipeline"""
    parser = argparse.ArgumentParser(prog='Automatic chord recognition', description='Transcribe audio signal')

    # create a default path 
    script_directory = os.path.dirname(os.path.abspath(__file__))
    default_path = os.path.join(script_directory, "mirdata")

    parser.add_argument('result_file',help="specify filename to save the results")
    parser.add_argument('--dataset_path',type=str,default=default_path,help="set path to datasets")
    parser.add_argument('--dataset',choices=['beatles','rwc_pop','rw','queen'],
                        default=['beatles','rwc_pop','rw','queen'],
                        help="select dataset or ommit to use all datasets"
                        )
    parser.add_argument('--transcriber', choices=['template', 'madmom'], default='template', 
                        help='select template based Recognition or use madmoms deep Chroma processor'
                        )
    parser.add_argument('--vocabulary', choices=['majmin', 'triads', 'triads_extended', 'sevenths'], 
                        default='majmin', help='select chord vocabulary'
                        )
    parser.add_argument('--eval_scheme', choices=['majmin','sevenths'], default='majmin',
                         help='Evaluation scheme'
                         )
    parser.add_argument('--source_separation', choices=['drums','vocals','both'], default=None,
                         help='Select source separation type'
                         )
    parser.add_argument('--prefilter', choices=[None, 'median', 'rp'], default='median', help='Select Prefilter type')
    parser.add_argument('--prefilter_length', type=int, default=7, help='Prefilter length')
    parser.add_argument('--embedding', type=int, default=25, help='Embedding value')
    parser.add_argument('--neighbors', type=int, default=50, help='Neighbours value')
    parser.add_argument('--postfilter', choices=[None, 'hmm', 'median'], default='hmm', help='select Postfilter type')
    parser.add_argument('--transition_prob', type=float, default=0.3, help='self-transition probability for a chord')
    parser.add_argument('--postfilter_length', type=int, default=4, help='Postfilter length')
    parser.add_argument('--display',choices=[True,False],default=False)
    args = parser.parse_args()

    # Convert Namespace to dictionary
    return vars(args)

def addDataset(group,dataset_name,data,metadata={}):
    """create a dataset entry in the given group of the hdf5 file"""
    try:
        dset = group.create_dataset(dataset_name,data=data)
    except ValueError as error: 
        print(f"Error while creating dataset {dataset_name}:{error}\moving on..")
        return
    for key,value in metadata.items():
        dset.attrs.create(str(key), value)
    return dset

def saveResults(group,track_id,scores={},track_data={}):
    """saves the results for a given track_id in hdf5 format.
     scores: a dictionary containing evaluation results
                e.g. scores = {"majmin":0.78,"seg_score":0.86}
     data:   a dictionary containing the datasets that need to be stored
                e.g. track_data = {"dataset_name": (mydata,mymetadata)}
    """
    # create a subgroup for the track_id
    subgroup = group.create_group(f"{track_id}")
    # store the evaluation scores as metadata of the group
    for eval_scheme,score in scores.items():
        subgroup.attrs.create(f"{eval_scheme}",score)

    for dataset_name,(data,metadata) in track_data.items():
        addDataset(subgroup,dataset_name,data,metadata)

if __name__ == "__main__":

    #### chord recognition experiment #### 
    params = parse_arguments()
    fpath = params["result_file"]+".hdf5"
    file = h5py.File(fpath, 'w')
    print(f"created file {fpath}")
    file.create_dataset('parameters', data=json.dumps(params))

    if isinstance(params["dataset"], list):
        dataset_list = params["dataset"]
    else:
        dataset_list = [params["dataset"]]     
       
    for datasetname in dataset_list:
        dataset = dataloader.Dataloader(datasetname,base_path=params["dataset_path"],source_separation=params["source_separation"])
        grp = file.create_group(datasetname)
        for fold in range(1,9):
            for track_id in tqdm(dataset.getExperimentSplits(fold),desc=f"fold {fold}/8"): 
                # Load audiofile and ground truth annotations
                name = dataset.getTitle(track_id)

                filepath,annotationpath = dataset[track_id]
                ref_intervals,ref_labels = utils.loadChordAnnotations(annotationpath)
                y = utils.loadAudiofile(filepath)
                gt = utils.loadChordAnnotations(annotationpath)

                # compute chromagram
                chroma_params = {"nCRP":33,"hop_length":2048,"fs":22050}
                chroma = features.crpChroma(y,nCRP=33,liftering=True,window=False)
                t_chroma = utils.timeVector(chroma.shape[1],hop_length=2048)
                

                # apply prefilter
                filter_type = params.get("prefilter")
                chroma_smoothed = features.applyPrefilter(t_chroma,chroma,filter_type,**params)

                # apply RMS thresholding
                rms_db = features.computeRMS(y)
                chroma_smoothed[:,rms_db < -60] = -1/12

                ## pattern matching ##         
                vocabulary = params.get("vocabulary")
                correlation,labels = features.computeCorrelation(chroma_smoothed,vocabulary)
                
                ## postfilter ##
                filter_type = params.get("postfilter")
                correlation_smoothed = features.applyPostfilter(correlation,labels,filter_type,**params)
                
                ## decode correlation matrix
                chord_sequence = [labels[i] for i in np.argmax(correlation_smoothed,axis=0)]    
                est_intervals,est_labels = utils.createChordIntervals(t_chroma,chord_sequence)   

                ## Evaluation ##
                eval_scheme = params.get("eval_scheme")
                score,seg_score = utils.evaluateTranscription(est_intervals,est_labels,ref_intervals,ref_labels,eval_scheme)
                

                # save results to file
                scores = {"majmin":score,"seg":seg_score}
                data = {"est_intervals":(est_intervals,{}),
                        "est_labels":(est_labels,{}),
                        "chroma":(chroma,chroma_params),
                        "chroma_enhanced":(chroma_smoothed,chroma_params)
                        }
                if params.get("display"):
                    plots.plotResults(t_chroma,chroma, chroma_smoothed, ref_intervals, ref_labels,
                                                est_intervals, est_labels)
                saveResults(grp,track_id,scores,data)

    file.close()
    print(f"DONE")