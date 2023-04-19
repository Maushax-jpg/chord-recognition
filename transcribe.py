
import dataloader
import chromagram
import joblib
import matplotlib.pyplot as plt


filepath = "/home/max/ET-TI/Masterarbeit/prototyping/data/beatles/audio/12_-_Let_It_Be/06_-_Let_It_Be.mp3"
modelpath = "/home/max/ET-TI/Masterarbeit/chord-recognition/hmm_model.pkl"

model = joblib.load(modelpath)
t,chroma = chromagram.getChroma(filepath,'madmom')
fig,ax = plt.subplots(figsize=(7,4))
chromagram.plotChroma(ax,chroma)


# chord_ix_predictions = h_markov_model.predict(chroma)
# index,predictions = postprocessing(chord_ix_predictions)
# ground_truth = dataloader.getBeatlesAnnotations(label_path)
# for row,col in ground_truth.iterrows():
#     if round(col['tstart']*10) < 100:
#         ax[1].text(round(col['tstart']*10),2,col['label'],color='k',horizontalalignment='center',rotation=90)
#         ax[1].vlines(round(col['tstart']*10),0,10,'k')
# for i,x in zip(index,predictions):
#     ax[1].text(i,1.5,x,color='r',horizontalalignment='center')
#     ax[1].vlines(i,0,10,'r')

plt.show()
