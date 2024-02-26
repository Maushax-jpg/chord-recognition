import os
import argparse
import dataloader
import utils
import features
import numpy as np
import h5py
from tqdm import tqdm
import pitchspace

# create paths
script_directory = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(script_directory, "mirdata")
cpss_model_path = os.path.join(script_directory, "models","cpss","cpss_sevenths.npy")
hmm_model_path = os.path.join(script_directory, "models","state_transitions","sevenths_30.npy")

def parse_arguments():
    """extract command line arguments in ordert to setup chord recognition pipeline"""
    parser = argparse.ArgumentParser(prog='Automatic chord recognition experiment', description='Transcribe audio signal')
    parser.add_argument('filename',help="specify filename to save the results")
    parser.add_argument('--source_separation', choices=["none",'drums','vocals','both'], default="none",
                         help='Select source separation type')
    parser.add_argument('--use_key_estimation', type=bool, default=True, help='use state transition matrix from data')
    parser.add_argument('--use_chord_statistics', type=bool, default=True, help='use state transition matrix from data')
    args = parser.parse_args()

    # Convert Namespace to dictionary
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

def postfilter(estimation_matrix):
    # postfiltering using HMM 
    if params.get("use_chord_statistics"):
        # load pretrained state transition probability matrix or use uniform state transition
        model_path = "/home/max/ET-TI/Masterarbeit/chord-recognition/models/state_transitions/sevenths_30.npy"
        A = np.load(model_path,allow_pickle=True)
    else:
        A = features.uniform_transition_matrix(0.2,len(labels)) 
    # compute quasi-normalized likelihood matrix   
    B_O = estimation_matrix / (np.sum(estimation_matrix,axis=0) + np.finfo(float).tiny) 
    C = np.ones((len(labels,))) * 1/len(labels)   # initial state probability matrix
    smoothed_estimation_matrix, _, _, _ = features.viterbi_log_likelihood(A, C, B_O)
    return smoothed_estimation_matrix


def transcribeCPSS(filepath,data,metadata,classifier):
    y = utils.loadAudiofile(filepath)
    chroma, pitchgram, pitchgram_cqt  = features.crpChroma(y,nCRP=33,window=True)
    t_chroma = utils.timeVector(chroma.shape[1],hop_length=2048)
    
    stable_regions, hcdf = classifier.computeStableRegions(chroma)

    # apply prefilter
    chroma = features.applyPrefilter(t_chroma,chroma,"median",N=7)

    data["chroma"] = (chroma,{"hop_length":2048,"fs":22050})
    data["hcdf"] = (hcdf,{"info":"harmonic change detection function"})
    data["pitchgram_cqt"] = (pitchgram_cqt,{"info":"unprocessed pitchgram derived from cqt"})

    # pattern matching with templates
    estimation_matrix_templates = features.computeCorrelation(chroma,templates,inner_product=False)

    # initialize matrices for estimation of stable regions
    templates_estimation_matrix = np.zeros_like(estimation_matrix_templates)
    templates_estimation_matrix[-1,:] = 1.0
    cpss_estimation_matrix = np.zeros_like(estimation_matrix_templates)
    cpss_estimation_matrix[-1,:] = 1.0

    # initialize matrix for "refined" template estimation with pitchspace
    refined_estimation_matrix = np.copy(estimation_matrix_templates)

    # classify stable regions
    for (i0, i1) in stable_regions:
        if params.get("use_key_estimation"):
            # find tonal center of stable regions with ~8 seconds context
            temp = np.average(chroma[:, np.maximum(0,i0-40):np.minimum(chroma.shape[1],i1+40)], axis=1)
            key_index = classifier.selectKey(temp)
        else:
            key_index = None
        # classify an average chromavector
        chroma_vector = np.average(chroma[:, i0:i1], axis=1).reshape((12,1))
        
        # use pitchspace to estimate chord in stable region
        index,label = classifier.classify(chroma_vector,key_index)
        cpss_estimation_matrix[index, i0:i1] = 1.0
        
        refined_estimation_matrix[:, i0:i1] = 0.0
        refined_estimation_matrix[index, i0:i1] = 1.0

        # use templates for comparison
        try:
            templates_estimation_matrix[:, i0:i1] = features.computeCorrelation(chroma_vector,templates,inner_product=False)
        except ValueError: # chroma_vector contains only NaN's
            continue
    # set No chord in regions with silence
    norm = np.sum(chroma,axis=0)
    refined_estimation_matrix[:,norm < 0.1] = 0.0
    refined_estimation_matrix[-1,norm < 0.1] = 1.0
    estimation_matrix_templates[:,norm < 0.1] = 0.0
    estimation_matrix_templates[-1,norm < 0.1] = 1.0

    # postfiltering using HMM 
    smoothed_estimation_matrix_templates = postfilter(estimation_matrix_templates)
    smoothed_refined_estimation_matrix = postfilter(refined_estimation_matrix)

    # create intervals and labels for all estimation matrices
    # stable regions transcribed with pitchspace
    chord_sequence = [classifier._labels[i] for i in np.argmax(cpss_estimation_matrix,axis=0)] 
    stable_cpss_intervals,stable_cpss_labels = utils.createChordIntervals(t_chroma,chord_sequence)

    # stable regions transcribed with templates
    chord_sequence = [classifier._labels[i] for i in np.argmax(templates_estimation_matrix,axis=0)] 
    stable_templates_intervals,stable_templates_labels = utils.createChordIntervals(t_chroma,chord_sequence)

    # chord estimation of total cpss system
    chord_sequence = [classifier._labels[i] for i in np.argmax(smoothed_refined_estimation_matrix,axis=0)] 
    est_cpss_intervals,est_cpss_labels = utils.createChordIntervals(t_chroma,chord_sequence)

    # chord estimation with template based approach
    chord_sequence = [classifier._labels[i] for i in np.argmax(smoothed_estimation_matrix_templates,axis=0)] 
    est_templates_intervals,est_templates_labels = utils.createChordIntervals(t_chroma,chord_sequence)


    data[f"stable_cpss_intervals"] = (stable_cpss_intervals,{"info":"stable region - chord intervals estimated with pitch space"})
    data[f"stable_cpss_labels"] = (stable_cpss_labels,{"info":"stable region - chord labels estimated with pitch space"})
    data[f"stable_templates_intervals"] = (stable_templates_intervals,{"info":"stable region - chord intervals estimated with templates"})
    data[f"stable_templates_labels"] = (stable_templates_labels,{"info":"stable region - chord labels estimated with templates"})
    data[f"est_cpss_intervals"] = (est_cpss_intervals,{"info":"estimated chord intervals"})
    data[f"est_cpss_labels"] = (est_cpss_labels,{"info":"estimated chord labels"})
    data[f"est_templates_intervals"] = (est_templates_intervals,{"info":"estimated chord intervals"})
    data[f"est_templates_labels"] = (est_templates_labels,{"info":"estimated chord labels"})
    # evaluation
    for model in ["stable_cpss","stable_templates","est_cpss","est_templates"]:
        for alphabet in ["majmin","sevenths"]:
            score,seg_score = utils.evaluateTranscription(data[f"{model}_intervals"][0],data[f"{model}_labels"][0],data["ref_intervals"][0],data["ref_labels"][0],alphabet)
            metadata[f"{model}_{alphabet}_score"] = score
            metadata[f"{model}_{alphabet}_segmentation"] = seg_score
            metadata[f"{model}_{alphabet}_f"] = round((2*score*seg_score)/(score+seg_score),2)
            print(alphabet,model,metadata[f"{model}_{alphabet}_f"])

if __name__ == "__main__":
    params = parse_arguments()
    file = h5py.File(params.pop("filename"), 'w')
    # save command line arguments as metadata
    for key,value in params.items():
        file.attrs.create(str(key), value)  

    classifier = pitchspace.CPSS_Classifier(cpss_model_path, "sevenths") 
    templates,labels = utils.createChordTemplates(template_type="sevenths") 

    for datasetname in dataloader.DATASETS:
        dataset = dataloader.Dataloader(datasetname,base_path=dataset_path,source_separation=params["source_separation"])
        file.create_group(datasetname)
        for track_id in tqdm(dataset.getTrackList(),desc=f"Dataset {datasetname}"): 
            metadata = {} # dictionary for additional track information
            data = {} # dictionary for track related data and metadata

            # Load audiofile and ground truth annotations
            metadata["name"] = dataset.getTitle(track_id)
            metadata["track_id"] = track_id
            filepath,annotationpath = dataset[track_id]
            ref_intervals,ref_labels = utils.loadChordAnnotations(annotationpath)
            data["ref_intervals"] = (ref_intervals,{"info":"ground truth intervals"})
            data["ref_labels"] = (ref_labels,{"info":"ground truth labels"})

            transcribeCPSS(filepath,data,metadata,classifier)
            # save results
            saveResults(file,track_id,metadata,data,datasetname)
    file.close()
    print(f"DONE")
