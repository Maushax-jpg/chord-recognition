import os
import argparse
import dataloader
import utils
import features
import numpy as np
import h5py
from tqdm import tqdm
import pitchspace

# create paths and specify used models
script_directory = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(script_directory, "mirdata")
cpss_model_path = os.path.join(script_directory, "models","cpss","cpss_sevenths.npy")
hmm_model_path = os.path.join(script_directory, "models","state_transitions","sevenths_30.npy")

def parse_arguments():
    """extract command line arguments in ordert to setup chord recognition pipeline"""
    parser = argparse.ArgumentParser(prog='Automatic chord recognition experiment', description='Transcribe audio signal')
    parser.add_argument('filename',help="specify filename to save the results")
    parser.add_argument('--source_separation', choices=["none",'drums','vocals','both'], default="drums",
                         help='Select source separation type')
    parser.add_argument('--N', type=int, default=11,
                         help='Select source separation type')
    parser.add_argument('--p', type=float, default=0.2,
                         help='Select source separation type')
    parser.add_argument('--use_key_estimation', type=bool, default=True, help='use center of effect for circle selection')
    parser.add_argument('--use_chord_statistics', type=bool, default=True, help='use state transition matrix from data')
    args = parser.parse_args()

    # Convert Namespace to dictionary
    return vars(args)

def postfilter(estimation_matrix):
    # postfiltering using HMM 
    if params.get("use_chord_statistics"):
        # load pretrained state transition probability matrix or use uniform state transition
        A = np.load(hmm_model_path,allow_pickle=True)
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

    # apply RMS thresholding 
    rms_db = features.computeRMS(y)
    mask = rms_db < -50
    chroma[:,mask] = utils.NO_CHORD

    # apply prefilter
    chroma = features.applyPrefilter(t_chroma,chroma,"median",N=params.get("N",11))

    # save data
    data["chroma"] = (chroma,{"hop_length":2048,"fs":22050})
    data["hcdf"] = (hcdf,{"info":"harmonic change detection function"})
    data["pitchgram_cqt"] = (pitchgram_cqt,{"info":"unprocessed pitchgram derived from cqt"})
    data["pitchgram"] = (pitchgram,{"info":"pitchgram after liftering"})

    # pattern matching with templates
    estimation_matrix_templates = features.computeCorrelation(chroma,templates)

    # initialize estimation matrices for estimation of stable regions with index of the No chord label
    estimation_matrix_stableregions_templates = np.zeros_like(estimation_matrix_templates)
    estimation_matrix_stableregions_cpss = np.zeros_like(estimation_matrix_templates)
    estimation_matrix_stableregions_templates[-1,:] = 1.0
    estimation_matrix_stableregions_cpss[-1,:] = 1.0

    # initialize matrix for "refined" template estimation with pitchspace
    estimation_matrix_cpss = np.copy(estimation_matrix_templates)

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
        chroma_vector = chroma_vector / np.sum(chroma_vector) # l1-normalization is necessary
        
        # use pitchspace to estimate chord in stable region
        index,_ = classifier.classify(chroma_vector,key_index)
        estimation_matrix_stableregions_cpss[index, i0:i1] = 1.0
        # set likelihood values for all chords to zero
        estimation_matrix_cpss[:, i0:i1] = 0.0
        # set likelihood of estimated index to 1 to enforce the chord
        estimation_matrix_cpss[index, i0:i1] = 1.0

        # as a comparison, estimate the chord with templates
        estimation_matrix_stableregions_templates[:, i0:i1] = features.computeCorrelation(chroma_vector,templates)

    # postfiltering using HMM 
    estimation_matrix_templates_smoothed = postfilter(estimation_matrix_templates)
    estimation_matrix_cpss_smoothed = postfilter(estimation_matrix_cpss)

    # create intervals and labels for all estimation matrices
    # stable regions transcribed with pitchspace
    chord_sequence = [classifier._labels[i] for i in np.argmax(estimation_matrix_stableregions_cpss,axis=0)] 
    intervals_stableregions_cpss,labels_stableregions_cpss = utils.createChordIntervals(t_chroma,chord_sequence)
    data[f"intervals_stableregions_cpss"] = (intervals_stableregions_cpss,{"info":"stable region - chord intervals estimated with pitch space"})
    data[f"labels_stableregions_cpss"] = (labels_stableregions_cpss,{"info":"stable region - chord labels estimated with pitch space"})
    
    # stable regions transcribed with templates
    chord_sequence = [classifier._labels[i] for i in np.argmax(estimation_matrix_stableregions_templates,axis=0)] 
    intervals_stableregions_templates,labels_stableregions_templates = utils.createChordIntervals(t_chroma,chord_sequence)
    data[f"intervals_stableregions_templates"] = (intervals_stableregions_templates,{"info":"stable region - chord intervals estimated with templates"})
    data[f"labels_stableregions_templates"] = (labels_stableregions_templates,{"info":"stable region - chord labels estimated with templates"})
   
    # chord estimation of cpss system
    chord_sequence = [classifier._labels[i] for i in np.argmax(estimation_matrix_cpss_smoothed,axis=0)] 
    intervals_cpss,labels_cpss = utils.createChordIntervals(t_chroma,chord_sequence)
    data[f"intervals_cpss"] = (intervals_cpss,{"info":"estimated chord intervals"})
    data[f"labels_cpss"] = (labels_cpss,{"info":"estimated chord labels"})

    # chord estimation with templates
    chord_sequence = [classifier._labels[i] for i in np.argmax(estimation_matrix_templates_smoothed,axis=0)] 
    intervals_templates,labels_templates = utils.createChordIntervals(t_chroma,chord_sequence)
    data[f"intervals_templates"] = (intervals_templates,{"info":"estimated chord intervals"})
    data[f"labels_templates"] = (labels_templates,{"info":"estimated chord labels"})
  
      # evaluation
    for model in ["stableregions_cpss","stableregions_templates","cpss","templates"]:
        for alphabet in ["majmin","sevenths"]:
            score,seg_score = utils.evaluateTranscription(data[f"intervals_{model}"][0],
                                                          data[f"labels_{model}"][0],
                                                          data["ref_intervals"][0],
                                                          data["ref_labels"][0],alphabet)
            metadata[f"{alphabet}_wcsr_{model}"] = score
            metadata[f"{alphabet}_q_{model}"] = seg_score
            metadata[f"{alphabet}_f_{model}"] = (2*score*seg_score) /(score+seg_score)
            print(alphabet,model,metadata[f"{alphabet}_f_{model}"])

if __name__ == "__main__":
    params = parse_arguments()
    file = h5py.File(params.pop("filename"), 'w')
    file.attrs.create("experiment","pitchspace_crp")
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
            utils.saveResults(file,track_id,metadata,data,datasetname)
    file.close()
    print(f"DONE")
