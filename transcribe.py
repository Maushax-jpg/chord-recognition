import HMM
import madmom
import dataloader
import utils
import matplotlib.pyplot as plt
import numpy as np

if __name__=='__main__':

    model_path = "/home/max/ET-TI/Masterarbeit/models/hmm_model.pkl"   
    model = HMM.load_model(model_path)

    dataset_path = "/home/max/ET-TI/Masterarbeit/datasets/beatles/"
    dataset_aligned = dataloader.BeatlesDataset(dataset_path,'madmom',beat_align=True)
    dataset = dataloader.BeatlesDataset(dataset_path,'madmom',beat_align=False)
    index = 103
    t_chroma,chroma,ref_intervals,ref_label = dataset_aligned[index]

    decoder = 'hmm'
    if decoder == 'hmm':
        # HMM prediction
        est_intervals,est_labels = model.predict(t_chroma,chroma)
    else:
        crp = madmom.features.chords.DeepChromaChordRecognitionProcessor()
        prediction = crp(chroma)
        # prediction ->  array([(tstart , stop, label), .. ,(tstart , stop, label)])
        est_intervals = np.array([[x[0],x[1]] for x in prediction])
        est_labels = [x[2] for x in prediction]

    # visualize results
    title = dataset_aligned.getTitle(index)
    audiopath,labelpath = dataset_aligned.getFilepaths(index)
    fig,ax = plt.subplots(3,1,figsize=(int(t_chroma[-1]),20))
    fig.tight_layout(pad=3)
    interval = (0,t_chroma[-1])
    utils.plotAudioWaveform(ax[0],audiopath,interval)
    utils.plotAnnotations(ax[0],ref_intervals,ref_label,interval)
    utils.plotPredictionResult(ax[0],est_intervals,est_labels,interval)
    utils.plotChroma(ax[1],chroma,interval)
    # plot chroma
    t_chroma,chroma,ref_intervals,ref_label = dataset[index]
    utils.plotChroma(ax[2],chroma,interval)
    plt.savefig(f"{title}.png")