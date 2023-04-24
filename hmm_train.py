import pandas as pd
import matplotlib.pyplot as plt
import dataloader
import numpy as np
import mir_eval
import HMM
import utils
def getMajMinSubset(df):
    """
    returns a subset containing major and minor chords from the Dataframe df.
    All tetrads or chords with altered scale degrees are filtered out.
    """
    majmin_df = pd.DataFrame(columns=df.columns)
    # create a subset of chromavectors, including 'no chord' symbol
    majmin_chords ={'N':'N'}
    enharmonic_dict = {'Db':'C#','Eb':'D#','Gb': 'F#','Ab': 'G#','Bb': 'A#'}

    for label in df['label'].unique():
        try:
            # parse chord label
            root,quality,scale,bass = mir_eval.chord.split(label)
            # skip all chords with altered scale notes or bass notes except 1,3,5
            if quality not in ['maj','min'] or scale or bass not in ['1','3','5']:
                continue
            # save major/minor chord with simplified label
            if root in enharmonic_dict:
                root = enharmonic_dict[root]
            majmin_chords[label] = mir_eval.chord.join(root,quality)
        except mir_eval.chord.InvalidChordException:
            print(f"invalid chord: {label}")
            continue
    for label in majmin_chords:
        temp = df[df['label']==label].copy()
        # rename chord with simplified label
        temp['label'] = majmin_chords[label]
        majmin_df = pd.concat([majmin_df,temp])
    # relabel
    return majmin_df


if __name__=='__main__':
    subset = 'majmin'
    path = "/home/max/ET-TI/Masterarbeit/datasets/beatles/beatles_librosa_chroma_df.pkl"
    df = pd.read_pickle(path)
    if subset == 'majmin':
        df = getMajMinSubset(df)

    # split in training / test data
    songs = df['title'].unique()
    states = df['label'].unique() 
    df = df.drop(columns=['time','title'])
    
    # maybe some cross validation?
    # use other datasets -> RWC? 
    ###############################
  
    model = HMM.HiddenMarkovModel(df,states)
    model.save("/home/max/ET-TI/Masterarbeit/models","hmm_model_librosa.pkl")
    
