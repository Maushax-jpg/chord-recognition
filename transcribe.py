import HMM
import madmom
import dataloader
import utils
import mir_eval
import matplotlib.pyplot as plt
import numpy as np



def createChordTemplates():
    """
    creates templates for Major and Minor chords
    return: list Chord_labels , ndarray templates (24x12)
    """
    pitch_class = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
    chord_labels = ["N"]
    templates = np.zeros((25,12),dtype=float)
    # major chords
    for i,x in enumerate(pitch_class):
        chord_labels.append(f"{x}:maj")
        templates[i+1,:] = np.roll(mir_eval.chord.quality_to_bitmap('maj'),i)
    # minor chords
    for i,x in enumerate(pitch_class):
        chord_labels.append(f"{x}:min")
        templates[i+12+1,:] = np.roll(mir_eval.chord.quality_to_bitmap('min'),i)
    return chord_labels,templates

def plotTemplates(labels,templates):
    fig,ax = plt.subplots()
    ax.imshow(templates)
    ax.set_xticks([0,2,4,5,7,9,11])
    ax.set_xticklabels(["C","D","E","F","G","A","B"])
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels);
    return fig

if __name__=='__main__':

    model_path = "/home/max/ET-TI/Masterarbeit/models/hmm_model.pkl"   
    model = HMM.load_model(model_path)

    dataset = dataloader.MIRDataset("beatles",use_deep_chroma=True,align_chroma=True)
    track_ids = dataset.getTrackIDs()

    y,t_chroma,chroma,ref_intervals,ref_labels = dataset[track_ids[104]]
    
    decoder = 'template'
    if decoder == 'hmm':
        # HMM prediction
        est_intervals,est_labels = model.predict(t_chroma,chroma)
    elif decoder == "dcp":
        crp = madmom.features.chords.DeepChromaChordRecognitionProcessor()
        prediction = crp(chroma)
        # prediction ->  array([(tstart , stop, label), .. ,(tstart , stop, label)])
        est_intervals = np.array([[x[0],x[1]] for x in prediction])
        est_labels = [x[2] for x in prediction]
    elif decoder == 'template':
        labels,templates = createChordTemplates()
        max_vals = np.max(chroma, axis=1)
        chroma_norm = chroma / max_vals[:,None]
        correlation = np.dot(templates,chroma_norm.T)
        est_labels_unfiltered = [labels[np.argmax(correlation[:,i])] for i in range(chroma.shape[0])]
        est_intervals = []
        est_labels = []
        t_start = 0
        for i,(label_t0,label_t1) in enumerate(zip(est_labels_unfiltered[:-1],est_labels_unfiltered[1:])):
            if label_t0 != label_t1:
                est_labels.append(label_t0)
                est_intervals.append([t_start,t_chroma[i]])
                t_start = t_chroma[i]
        est_intervals = np.array(est_intervals)


    # visualize results
    fig,ax = plt.subplots(2,1,figsize=(10,10))
    interval = (0,15)
    utils.plotAudioWaveform(ax[0],y,interval)
    utils.plotAnnotations(ax[0],ref_intervals,ref_labels,interval)
    utils.plotPredictionResult(ax[0],est_intervals,est_labels,interval)
    utils.plotChroma(ax[1],chroma,interval)
    plt.show()