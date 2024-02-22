import os
import argparse
import dataloader
import utils
import features
import numpy as np
import h5py
from tqdm import tqdm
import pitchspace

# create a default path 
script_directory = os.path.dirname(os.path.abspath(__file__))
default_path = os.path.join(script_directory, "mirdata")

def parse_arguments():
    """extract command line arguments in ordert to setup chord recognition pipeline"""
    global default_path
    parser = argparse.ArgumentParser(prog='Automatic chord recognition experiment', description='Transcribe audio signal')
    parser.add_argument('filename',help="specify filename to save the results")
    parser.add_argument('--dataset_path',type=str,default=default_path,help="set path to datasets")
    parser.add_argument('--dataset',choices=['beatles','rwc_pop','rw','queen'],
                        default=['beatles','rwc_pop','rw','queen'],
                        help="select dataset or ommit to use all datasets")
    parser.add_argument('--transcriber', choices=['template', 'madmom','cpss'], default='template', 
                        help='select template based Recognition or use madmoms deep Chroma processor')
    parser.add_argument('--chroma_type', choices=['crp','dcp'], default='dcp', 
                        help='select chromagram type')
    parser.add_argument('--use_chord_statistics', type=bool, default=True, help='use state transition matrix from data')
    parser.add_argument('--transition_prob', type=float, default=0.2, help='self-transition probability for a chord')
    args = parser.parse_args()

    # Convert Namespace to dictionary
    return vars(args)

def transcribeDeepChroma(filepath,fold,data,metadata):
    import madmom
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

def transcribeTemplate(filepath,split,data,metadata,params):
    y = utils.loadAudiofile(filepath)
    # compute chromagram
    chroma_params = {"hop_length":2205,"fs":22050,"type":"dcp"}
    chroma = features.deepChroma(filepath,split)
    t_chroma = utils.timeVector(chroma.shape[1],hop_length=chroma_params["hop_length"])
    chroma = features.applyPrefilter(t_chroma,chroma,"median",N=7)
    chroma_norm =  np.sum(chroma,axis=0)
    data["chroma"] = (chroma,chroma_params)

    for alphabet in ["majmin","sevenths"]:
        ## pattern matching ##         
        correlation,labels = features.computeCorrelation(chroma,inner_product=False,template_type=alphabet)
        
        # apply RMS thresholding to enforce No chord
        mask = chroma_norm < 0.3 # 
        correlation[:,mask] = 0.0
        correlation[-1,mask] = 1.0    

        ## postfiltering using HMM ##
        # load pretrained state transition probability matrix or use uniform state transition
        if params.get("use_chord_statistics"):
            global script_directory
            model_path = os.path.join(script_directory,"models","state_transitions",f"{alphabet}_20.npy")
            A = np.load(model_path,allow_pickle=True)
        else:
            p = params.get("transition_prob",0.1)
            A = features.uniform_transition_matrix(p,len(labels)) 
            
        B_O = correlation / (np.sum(correlation,axis=0) + np.finfo(float).tiny) # likelyhood matrix
        C = np.ones((len(labels,))) * 1/len(labels)   # initial state probability matrix
        correlation_smoothed, _, _, _ = features.viterbi_log_likelihood(A, C, B_O)

        ## decode correlation matrix ##
        if alphabet == "majmin":
            chord_sequence = [labels[i] for i in np.argmax(correlation_smoothed,axis=0)]   
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

def transcribeCPSS(filepath,split,data,metadata,classifier):
    y = utils.loadAudiofile(filepath)
    chroma = features.deepChroma(filepath,split)
    t_chroma = utils.timeVector(chroma.shape[1],hop_length=2205)
    hcdf = pitchspace.computeHCDF(chroma,prefilter_length=3,use_cpss=False)
    
    chroma_norm =  np.sum(chroma,axis=0)
    mask = chroma_norm < 0.1
    hcdf[mask] = 1
    ## thresholding parameters
    threshold = 0.3
    min_distance = 1

    ## Find stable regions in the chromagram
    gate = np.zeros_like(hcdf)
    gate[hcdf < threshold] = 1

    stable_regions = []
    start_index = 0

    for i, value in enumerate(gate):
        if value == 1:
            if start_index is None:
                start_index = i
        elif start_index is not None:
            # check if the interval is long enough
            if i - start_index > min_distance: 
                stable_regions.append((start_index, i-1))
            start_index = None
    # check if last index is still ongoing
    if start_index is not None:
        stable_regions.append((start_index, len(gate) - 1))

    # apply prefilter
    chroma = features.applyPrefilter(t_chroma,chroma,"median",N=7)

    data["chroma"] = (chroma,{"hop_length":2205,"fs":22050}) # DCP
    data["hcdf"] = (hcdf,{"info":"harmonic change detection function"})
    data["gate"] = (gate,{"info":"gating function"})
    correlation, labels = features.computeCorrelation(chroma,inner_product=False,template_type="sevenths")

    # save the estimation of stable regions only (for visualization)
    corr_stable = np.zeros_like(correlation)
    corr_stable[-1,:] = 1.0

    # classify stable regions witch tonal pitch space
    for (i0, i1) in stable_regions:
        if True:
            # find tonal center of stable regions with ~8 seconds context
            i_a = np.maximum(0,i0-40)
            i_b = np.minimum(t_chroma.shape[0],i1+40)
        else:
            i_a = i0
            i_b = i1
        chroma_regional = np.average(chroma[:, i_a:i_b],axis=1).reshape((12,1))
        ic_energy = np.zeros((12,))  
        for i in range(12):
            # neglect all chromabins that are not part of the diatonic circle -> key_related_chroma
            pitch_classes = np.roll([1,0,1,0,1,1,0,1,0,1,0,1],i)
            key_related_chroma = np.multiply(chroma_regional, np.reshape(pitch_classes, (12,1))) 
            ic_energy[i] = np.sum(features.computeIntervalCategories(key_related_chroma), axis=0)
        key_index = np.argmax(ic_energy)
        if ic_energy[key_index] > 0.0: # check if there is some information on intervals available
            # classify the average chromavector of the stable region in the pitch system
            index = classifier.classify(np.average(chroma[:, i0:i1],axis=1),key_index)
        else:
            index = None
        if index is not None:
            correlation[:, i0:i1] = 0.0
            correlation[index, i0:i1] = 1.0
            corr_stable[index, i0:i1] = 1.0

    # decode estimation of stable regions
    chord_sequence = [labels[i] for i in np.argmax(corr_stable,axis=0)] 
    est_intervals_cpss,est_labels_cpss = utils.createChordIntervals(t_chroma,chord_sequence)

    ## viterbi smoothing 
    # load pretrained state transition probability matrix or use uniform state transition
    if params.get("use_chord_statistics"):
        global script_directory
        model_path = os.path.join(script_directory,"models","state_transitions",f"sevenths_30.npy")
        A = np.load(model_path,allow_pickle=True)
    else:
        p = params.get("transition_prob",0.1)
        A = features.uniform_transition_matrix(p,len(labels)) 

    # apply RMS thresholding, force No chord if RMS is below threshold
    correlation[:,mask] = 0.0
    correlation[-1,mask] = 1.0    

    B_O = correlation / (np.sum(correlation,axis=0) + np.finfo(float).tiny) # likelyhood matrix
    C = np.ones((len(labels,))) * 1/len(labels)   # initial state probability matrix
    correlation_smoothed, _, _, _ = features.viterbi_log_likelihood(A, C, B_O)

    chord_sequence = [labels[i] for i in np.argmax(correlation_smoothed,axis=0)] 
    est_intervals,est_labels = utils.createChordIntervals(t_chroma,chord_sequence)
    
    data[f"intervals_cpss"] = (est_intervals_cpss,{"info":"estimated chord intervals with pitch space"})
    data[f"labels_cpss"] = (est_labels_cpss,{"info":"estimated chord labels with pitch space"})
    data[f"refined_intervals"] = (est_intervals,{"info":"estimated chord intervals"})
    data[f"refined_labels"] = (est_labels,{"info":"estimated chord labels"})

    # save results
    for alphabet in ["majmin","sevenths"]:
        try:
            score,seg_score = utils.evaluateTranscription(est_intervals,est_labels,data["ref_intervals"][0],data["ref_labels"][0],alphabet)
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

    if isinstance(params["dataset"], list):
        dataset_list = params["dataset"]
    else:
        dataset_list = [params["dataset"]]     

    if params["transcriber"] == "cpss":
        modelpath = os.path.join(script_directory, "models","cpss")
        classifier = pitchspace.CPSS_Classifier("sevenths",modelpath) 

    for datasetname in dataset_list:
        dataset = dataloader.Dataloader(datasetname,base_path=dataset_path,source_separation="none")
        file.create_group(datasetname)
        for split in range(1,9):
            for track_id in tqdm(dataset.getExperimentSplits(split),desc=f"split {split}/8"): 
                metadata = {} # dictionary for additional track information
                data = {} # dictionary for track related data and metadata

                # Load audiofile and ground truth annotations
                metadata["name"] = dataset.getTitle(track_id)
                metadata["track_id"] = track_id
                metadata["split"] = split
                filepath,annotationpath = dataset[track_id]
                ref_intervals,ref_labels = utils.loadChordAnnotations(annotationpath)
                data["ref_intervals"] = (ref_intervals,{"info":"ground truth intervals"})
                data["ref_labels"] = (ref_labels,{"info":"ground truth labels"})

                if params["transcriber"] == "template":
                    transcribeTemplate(filepath,split,data,metadata,params)
                elif params["transcriber"] == "madmom":
                    transcribeDeepChroma(filepath,split,data,metadata)
                elif params["transcriber"] == "cpss":
                    transcribeCPSS(filepath,split,data,metadata,classifier)
                # save results
                saveResults(file,track_id,metadata,data,datasetname)
                break
            break
    file.close()
    print(f"DONE")
