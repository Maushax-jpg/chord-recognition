import os
import argparse
import dataloader
import utils
import features
import numpy as np
import h5py
from tqdm import tqdm
import madmom
import pitchspace

def parse_arguments():
    """extract command line arguments in ordert to setup chord recognition pipeline"""
    parser = argparse.ArgumentParser(prog='Automatic chord recognition experiment', description='Transcribe audio signal')

    # create a default path 
    script_directory = os.path.dirname(os.path.abspath(__file__))
    default_path = os.path.join(script_directory, "mirdata")

    parser.add_argument('filename',help="specify filename to save the results")
    parser.add_argument('--dataset_path',type=str,default=default_path,help="set path to datasets")
    parser.add_argument('--dataset',choices=['beatles','rwc_pop','rw','queen'],
                        default=['beatles','rwc_pop','rw','queen'],
                        help="select dataset or ommit to use all datasets")
    parser.add_argument('--transcriber', choices=['template', 'madmom','cpss'], default='template', 
                        help='select template based Recognition or use madmoms deep Chroma processor')
    parser.add_argument('--chroma_type', choices=['CRP','Clog','CQT'], default='CRP', 
                        help='select chromagram type')
    parser.add_argument('--source_separation', choices=["none",'drums','vocals','both'], default="none",
                         help='Select source separation type')
    parser.add_argument('--prefilter', choices=[None, 'median', 'rp'], default='median', help='Select Prefilter type')
    parser.add_argument('--prefilter_length', type=int, default=7, help='Prefilter length')
    parser.add_argument('--embedding', type=int, default=25, help='Embedding value')
    parser.add_argument('--neighbors', type=int, default=50, help='Neighbours value')
    parser.add_argument('--postfilter', choices=[None, 'hmm', 'median'], default='hmm', help='select Postfilter type')
    parser.add_argument('--use_chord_statistics', type=bool, default=True, help='use state transition matrix from data')
    parser.add_argument('--transition_prob', type=float, default=0.2, help='self-transition probability for a chord')
    parser.add_argument('--postfilter_length', type=int, default=4, help='Postfilter length')
    args = parser.parse_args()

    # Convert Namespace to dictionary
    return vars(args)

def transcribeDeepChroma(filepath,fold,data,metadata):
    chroma = features.deepChroma(filepath,fold)
    chromadata = {"fs":22050,"hop_length":2205}
    chord_processor = madmom.features.chords.DeepChromaChordRecognitionProcessor()
    estimations = chord_processor(chroma.T)
    intervals = np.array([(x[0],x[1]) for x in estimations])
    labels = [x[2] for x in estimations]
    ## Evaluation ##
    for alphabet in ["majmin","sevenths"]:
        score,seg_score = utils.evaluateTranscription(intervals,labels,data["ref_intervals"][0],data["ref_labels"][0],alphabet)
        metadata[f"{alphabet}_score"] = score
        metadata[f"{alphabet}_segmentation"] = seg_score
        metadata[f"{alphabet}_f"] = round((2*score*seg_score)/(score+seg_score),2)
        data["chroma"] = (chroma,chromadata)
        data[f"{alphabet}_intervals"] = (intervals,{"info":"estimated chord intervals"}) # madmom only provides major/minor chord transcription
        data[f"{alphabet}_labels"] = (labels,{"info":"estimated chord labels"})
    return intervals,labels


def saveResults(file,name,track_metadata,datasets,parent_group=None):
    """saves the results for a given track_id in hdf5 format
    file: hdf5 file handle 
    track_id: the track ID of the song 
    track_metadata: additional attributes for the track
    datasets:   a dictionary containing the datasets that need to be stored
                e.g. track_data = {"dataset_1": (data,metadata)}
    """
    # create a subgroup for the track_id
    if parent_group is not None:
        subgroup = file.create_group(f"{parent_group}/{name}")
    else:
        subgroup = file.create_group(f"{name}")
    for key,value in track_metadata.items():
        subgroup.attrs.create(str(key), value)
    
    # store the datasets 
    for dataset_name,(data,metadata) in datasets.items():
        try:
            dset = subgroup.create_dataset(dataset_name,data=data)
        except ValueError as error: 
            print(f"Error while creating dataset {dataset_name}:{error}\moving on..")
            continue
        for key,value in metadata.items():
            dset.attrs.create(str(key), value)

def transcribeTemplate(y,data,metadata,params):
    # compute chromagram
    chroma_type = params.get("chroma_type")

    chroma_params = {"hop_length":2048,"fs":22050,"type":chroma_type}
    if chroma_type == "CRP":
        chroma_params["nCRP"] = 33
        chroma_params["eta"] = 100
        chroma_params["window"] = True
        chroma, C = features.crpChroma(y,nCRP=33,eta=100,liftering=True,window=True)
    elif chroma_type == "Clog":
        chroma_params["eta"] = 100
        chroma_params["window"] = True
        chroma, C = features.crpChroma(y,eta=100,liftering=False,window=True)
    else:
        chroma, C = features.crpChroma(y,liftering=False,compression=False,window=False)

    t_chroma = utils.timeVector(chroma.shape[1],hop_length=2048)
    data["chroma"] = (chroma,chroma_params)
    data["cqt"] = (C,{})
    # apply prefilter
    filter_type = params.get("prefilter")
    chroma_smoothed = features.applyPrefilter(t_chroma,chroma,filter_type,**params)
    
    # apply RMS thresholding
    rms_db = features.computeRMS(y)
    chroma_smoothed[:,rms_db < -50] = -1/12
    data["chroma_prefiltered"] = (chroma_smoothed,chroma_params)

    for alphabet in ["majmin","sevenths"]:
        ## pattern matching ##         
        correlation,labels = features.computeCorrelation(chroma_smoothed,inner_product=True,template_type=alphabet)
        data["correlation"] = (correlation,{"info":"cross-correlation with templates"})
        ## postfilter ##
        filter_type = params.get("postfilter")
        correlation_smoothed = features.applyPostfilter(correlation,labels,filter_type,**params)
        
        ## decode correlation matrix ##
        if alphabet == "majmin":
            chord_sequence = [labels[i] for i in np.argmax(correlation_smoothed[:24,:],axis=0)]   
        else:
            chord_sequence = [labels[i] for i in np.argmax(correlation_smoothed,axis=0)]  
        intervals,labels = utils.createChordIntervals(t_chroma,chord_sequence)   
        data[f"{alphabet}_intervals"] = (intervals,{"info":"estimated chord intervals"})
        data[f"{alphabet}_labels"] = (labels,{"info":"estimated chord labels"})

        ## Evaluation ##
        score,seg_score = utils.evaluateTranscription(intervals,labels,data["ref_intervals"][0],data["ref_labels"][0],alphabet)
        metadata[f"{alphabet}_score"] = score
        metadata[f"{alphabet}_segmentation"] = seg_score
        metadata[f"{alphabet}_f"] = round((2*score*seg_score)/(score+seg_score),2)
    return 

def transcribeCPSS(filepath,data,metadata,classifier):
    y = utils.loadAudiofile(filepath)
    chroma = features.crpChroma(y,nCRP=33)
    t_chroma = utils.timeVector(chroma.shape[1],hop_length=2048)
    hcdf = pitchspace.computeHCDF(chroma,prefilter_length=7,use_cpss=False)
    ## thresholding
    threshold = 0.3
    min_distance = 2
    gate = np.zeros_like(hcdf)
    gate[hcdf < threshold] = 1
    chroma_indices = []
    start_index = 0

    for i, value in enumerate(gate):
        if value == 1:
            if start_index is None:
                start_index = i
        elif start_index is not None:
            # check if the interval is long enough
            if i - start_index > min_distance: 
                chroma_indices.append((start_index, i-1))
            start_index = None
    # check if last index is still ongoing
    if start_index is not None:
        chroma_indices.append((start_index, len(gate) - 1))

    # apply prefilter
    filter_type = params.get("prefilter")
    chroma = features.applyPrefilter(t_chroma,chroma,filter_type,**params)
    data["chroma"] = (chroma,{"hop_length":2048,"fs":22050})

    # No chord until first stable region
    est_cpss_intervals = [[0,t_chroma[chroma_indices[0][0]]]]
    est_cpss_labels = ["N"]

    for x in chroma_indices:
        chroma_median = np.median(chroma[:,x[0]:x[1]],axis=1)
        label, _ = classifier.classify(chroma_median)
        
        t_start = t_chroma[x[0]]
        t_stop = t_chroma[x[1]]
        # check time difference to last stable region
        dt = t_start - est_cpss_intervals[-1][1]
        if dt < 1:
            # adjust intervals
            t_start -= dt/2
            est_cpss_intervals[-1][1] += dt/2
        else:
            # add no chord for the unstable region in between the last and current region
            est_cpss_labels.append("N")
            est_cpss_intervals.append([est_cpss_intervals[-1][1],t_start])
        # add current region
        est_cpss_intervals.append([t_start, t_stop])
        est_cpss_labels.append(label)
        
    est_cpss_intervals = np.array(est_cpss_intervals)
    # save results
    for alphabet in ["majmin","sevenths"]:
        data[f"{alphabet}_intervals"] = (est_cpss_intervals,{"info":"estimated chord intervals"})
        data[f"{alphabet}_labels"] = (est_cpss_labels,{"info":"estimated chord labels"})
        try:
            score,seg_score = utils.evaluateTranscription(est_cpss_intervals,est_cpss_labels,data["ref_intervals"][0],data["ref_labels"][0],alphabet)
        except Exception as e:
            print(e)
            score = 0.01
            seg_score = 0.01
        metadata[f"{alphabet}_score"] = score
        metadata[f"{alphabet}_segmentation"] = seg_score
        metadata[f"{alphabet}_f"] = round((2*score*seg_score)/(score+seg_score),2)

if __name__ == "__main__":
    params = parse_arguments()
    file = h5py.File(params.pop("filename"), 'w')
    dataset_path = params.pop("dataset_path")
    # save command line arguments as metadata
    for key,value in params.items():
        if value is None:
            value = "None"
        file.attrs.create(str(key), value)

    if params.get("transcriber") == "cpss":
        classifier = pitchspace.Classifier()

    if isinstance(params["dataset"], list):
        dataset_list = params["dataset"]
    else:
        dataset_list = [params["dataset"]]     

    for datasetname in dataset_list:
        dataset = dataloader.Dataloader(datasetname,base_path=dataset_path,source_separation=params["source_separation"])
        file.create_group(datasetname)
        for fold in range(1,9):
            for track_id in tqdm(dataset.getExperimentSplits(fold),desc=f"fold {fold}/8"): 
                metadata = {} # dictionary for additional track information
                data = {} # dictionary for track related data and metadata

                # Load audiofile and ground truth annotations
                metadata["name"] = dataset.getTitle(track_id)
                filepath,annotationpath = dataset[track_id]
                ref_intervals,ref_labels = utils.loadChordAnnotations(annotationpath)
                data["ref_intervals"] = (ref_intervals,{"info":"ground truth intervals"})
                data["ref_labels"] = (ref_labels,{"info":"ground truth labels"})

                if params["transcriber"] == "template":
                    y = utils.loadAudiofile(filepath)
                    transcribeTemplate(y,data,metadata,params)
                elif params["transcriber"] == "madmom":
                    transcribeDeepChroma(filepath,fold,data,metadata)
                elif params["transcriber"] == "cpss":
                    transcribeCPSS(filepath,data,metadata,classifier)
                # save results
                saveResults(file,track_id,metadata,data,datasetname)

    file.close()
    print(f"DONE")
