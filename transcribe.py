import HMM
import pandas as pd
import dataloader
chroma_df = pd.read_pickle("/home/max/ET-TI/Masterarbeit/prototyping/data/beatles/beatles_chroma_majmin.pkl")  
# TODO rename label '' to 'N'
labels = chroma_df['label'].unique()
labels.sort()
hmm = HMM.HiddenMarkovModel(chroma_df,labels)

dataset = dataloader.BeatlesDataset()
title,chroma,annotations = dataset[25]
intervals,chords = hmm.predict(chroma)
for t,label in zip(intervals,chords):
    print(t,label)