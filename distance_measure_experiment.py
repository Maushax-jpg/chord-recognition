import os
import argparse
import utils
import dataloader
import features
import numpy as np
import h5py
from tqdm import tqdm

# create a default path 
script_directory = os.path.dirname(os.path.abspath(__file__))
default_path = os.path.join(script_directory, "mirdata")

def parse_arguments():
    """extract command line arguments in ordert to setup chord recognition pipeline"""
    global default_path
    parser = argparse.ArgumentParser(prog='Automatic chord recognition experiment', description='Transcribe audio signal')
    parser.add_argument('filename',help="specify filename to save the results")
    parser.add_argument('--source_separation', choices=["none",'drums','vocals','both'], default="none",
                         help='Select source separation type')
    args = parser.parse_args()
    return vars(args)

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

def postfilter(distance_matrix,labels):
    ## postfiltering using HMM ##
    A = features.uniform_transition_matrix(0.2,len(labels)) # state transition probability matrix
    B = distance_matrix / (np.sum(distance_matrix,axis=0) + features.EPS) # likelyhood matrix
    C = np.ones((len(labels,))) * 1/len(labels)   # initial state probability matrix
    distance_matrix_smoothed, _, _, _ = features.viterbi_log_likelihood(A, C, B)
    return distance_matrix_smoothed

def transcribeTemplate(filepath,data,metadata,params):
    y = utils.loadAudiofile(filepath)

    chroma_params = {"hop_length":2048,"fs":22050,"type":"crp"}
    chroma_params["nCRP"] = 33
    chroma_params["eta"] = 100
    chroma_params["window"] = False
    chroma, pitchgram,pitchgram_cqt = features.crpChroma(y,nCRP=chroma_params["nCRP"],
                                                         eta=chroma_params["eta"],
                                                         liftering=True,window=chroma_params["window"])

    t_chroma = utils.timeVector(chroma.shape[1],hop_length=chroma_params["hop_length"])
    data["pitchgram_cqt"] = (pitchgram_cqt,{"info":"unprocessed pitchgram derived from cqt"})
    
    # apply RMS thresholding
    rms_db = features.computeRMS(y)
    mask = rms_db < -50
    chroma[:,mask] = utils.NO_CHORD

    # apply prefilter
    chroma_smoothed = features.applyPrefilter(t_chroma,chroma,"median",N=11)
    data["chroma"] = (chroma_smoothed,chroma_params)

    templates,labels = utils.createChordTemplates(template_type="sevenths") 

    for alphabet in ["majmin","sevenths"]:
        # pattern matching with sample correlation 
        correlation = features.computeCorrelation(chroma_smoothed,templates,inner_product=False)

        # pattern matching with inner product
        inner_product = features.computeCorrelation(chroma_smoothed,templates,inner_product=True)
        inner_product[:,mask] = 0.0 # reset correlation to enforce No chord in silence regions
        inner_product[-1,mask] = 1.0

        # HMM 
        correlation_smoothed = postfilter(correlation,labels)        
        inner_product_smoothed = postfilter(inner_product,labels)

        # decode correlation matrix 
        chord_sequence = [labels[i] for i in np.argmax(correlation_smoothed,axis=0)]  
        est_intervals, est_labels = utils.createChordIntervals(t_chroma,chord_sequence)   
        data[f"{alphabet}_intervals_correlation"] = (est_intervals,{"info":"estimated chord intervals"})
        data[f"{alphabet}_labels_correlation"] = (est_labels,{"info":"estimated chord labels"})
        # Evaluation 
        score,seg_score = utils.evaluateTranscription(est_intervals,est_labels,data["ref_intervals"][0],data["ref_labels"][0],alphabet)
        metadata[f"{alphabet}_wscr_correlation"] = score
        metadata[f"{alphabet}_q_correlation"] = seg_score
        metadata[f"{alphabet}_f_correlation"] = round((2*score*seg_score)/(score+seg_score),2)

        ## decode matrix of inner products 
        chord_sequence = [labels[i] for i in np.argmax(inner_product_smoothed,axis=0)]  
        est_intervals,est_labels = utils.createChordIntervals(t_chroma,chord_sequence)   
        data[f"{alphabet}_intervals_inner_product"] = (est_intervals,{"info":"estimated chord intervals"})
        data[f"{alphabet}_labels_inner_product"] = (est_labels,{"info":"estimated chord labels"})

        # Evaluation 
        score,seg_score = utils.evaluateTranscription(est_intervals,est_labels,data["ref_intervals"][0],data["ref_labels"][0],alphabet)
        metadata[f"{alphabet}_wcsr_inner_product"] = score
        metadata[f"{alphabet}_q_inner_product"] = seg_score
        metadata[f"{alphabet}_f_inner_product"] = round((2*score*seg_score)/(score+seg_score),2)
        print(alphabet,"inner: ",metadata[f"{alphabet}_f_inner_product"], "corr: ",metadata[f"{alphabet}_f_correlation"] )
    return 

if __name__ == "__main__":
    params = parse_arguments()
    # specify used parameters
    params["N"] = 11
    params["p"] = 0.2
    file = h5py.File(params.pop("filename"), 'w')
    # save command line arguments as metadata
    for key,value in params.items():
        file.attrs.create(str(key), value)
    print(f"Starting Transcription / {params['source_separation']}")
    for dset_name in dataloader.DATASETS:
        dataset = dataloader.Dataloader(dset_name,base_path=default_path, source_separation=params["source_separation"])
        file.create_group(dset_name)
        for track_id in tqdm(dataset.getTrackList(),desc=f"{dset_name}"): 
            metadata = {} # dictionary for additional track information
            data = {} # dictionary for track related data and metadata
            # Load audiofile and ground truth annotations
            metadata["name"] = dataset.getTitle(track_id)
            metadata["track_id"] = track_id
            filepath,annotationpath = dataset[track_id]
            ref_intervals,ref_labels = utils.loadChordAnnotations(annotationpath)
            data["ref_intervals"] = (ref_intervals,{"info":"ground truth intervals"})
            data["ref_labels"] = (ref_labels,{"info":"ground truth labels"})

            transcribeTemplate(filepath,data,metadata,params)
            # save results
            saveResults(file,track_id,metadata,data,dset_name)
    file.close()
    print(f"DONE")
