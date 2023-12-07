import os
import argparse
import dataloader
import utils
import features

def parse_arguments():
    """extract command line arguments in ordert to setup chord recognition pipeline"""
    parser = argparse.ArgumentParser(prog='Automatic chord recognition', description='Transcribe audio signal')

    # create a default path 
    script_directory = os.path.dirname(os.path.abspath(__file__))
    default_path = os.path.join(script_directory, "mirdata")

    parser.add_argument('--filepath',type=str,default=None,help="set path to audiofile or ommit to transcribe a dataset")
    parser.add_argument('--dataset_path',type=str,default=default_path,help="set path to datasets")
    parser.add_argument('--dataset',choices=['beatles','rwc_pop','rw','queen'],
                        default=['beatles','rwc_pop','rw','queen'],
                        help="select dataset or ommit to use all datasets"
                        )
    parser.add_argument('--transcriber', choices=['template', 'madmom'], default='template', 
                        help='select template based Recognition or use madmoms deep Chroma processor'
                        )
    parser.add_argument('--vocabulary', choices=['majmin', 'triads', 'triads_extended', 'majmin_sevenths'], 
                        default='majmin', help='select chord vocabulary'
                        )
    parser.add_argument('--eval_scheme', choices=['majmin','majmin_sevenths'], default='majmin',
                         help='Evaluation scheme'
                         )
    parser.add_argument('--source_separation', choices=['drums','vocals','both'], default=None,
                         help='Select source separation type'
                         )
    parser.add_argument('--prefilter', choices=[None, 'median', 'rp'], default='median', help='Select Prefilter type')
    parser.add_argument('--median_length', type=int, default=7, help='Prefilter length')
    parser.add_argument('--rp_embedding', type=int, default=25, help='Embedding value')
    parser.add_argument('--rp_neighbours', type=int, default=50, help='Neighbours value')
    parser.add_argument('--postfilter', choices=[None, 'hmm', 'median'], default='hmm', help='select Postfilter type')
    parser.add_argument('--transition_prob', type=float, default=0.3, help='self-transition probability for a chord')
    parser.add_argument('--postfilter_length', type=int, default=4, help='Postfilter length')

    args = parser.parse_args()

    # Convert Namespace to dictionary
    return vars(args)

if __name__ == "__main__":
    params = parse_arguments()
    
    filepath = params["filepath"]
    if filepath is not None:
        # transcribe song
        print(filepath)
        quit()


    # Experimental pipeline
    from tqdm import tqdm
    print(f"Starting Transcription!")
    for name in params["dataset"]:
        dataset = dataloader.Dataloader(name,base_path=params["dataset_path"],source_separation=params["source_separation"])
        print(f"Dataset: {name}")
        for fold in range(1,9):
            for track_id in tqdm(dataset.getExperimentSplits(fold),desc=f"fold {fold}/8"):  
                
                # Load audiofile
                name = dataset.getTitle(track_id)
                filepath,annotationpath = dataset[track_id]
                ref_intervals,ref_labels = utils.loadChordAnnotations(annotationpath)
                y = utils.loadAudiofile(filepath)

                # compute chromagram
                chroma = features.crpChroma(y)
                t_chroma = utils.timeVector(chroma.shape[1],hop_length=2048)
                #est_intervals, est_labels = transcribeChromagram(t_chroma,chroma,**parameters)    