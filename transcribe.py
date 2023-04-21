import HMM
import pandas as pd
import dataloader

# load model 
chroma_df = pd.read_pickle("/home/max/ET-TI/Masterarbeit/prototyping/data/beatles/beatles_chroma_majmin.pkl")  
# TODO rename label '' to 'N'
labels = chroma_df['label'].unique()
labels.sort()
hmm = HMM.HiddenMarkovModel(chroma_df,labels)

dataset = dataloader.BeatlesDataset()
title,chroma,annotations = dataset[105]
t,features = chroma
intervals,chords = hmm.predict(t,features)
print(len(intervals[0]))
#transcribtion_df = pd.DataFrame(,columns=['t_start','t_stop','chords'])
#print(transcribtion_df.head())
