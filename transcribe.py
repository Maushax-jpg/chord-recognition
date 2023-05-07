import HMM
import madmom
import dataloader
import evaluate
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
    chord_labels = []
    templates = np.zeros((24,12),dtype=float)
    # major chords
    for i,x in enumerate(pitch_class):
        chord_labels.append(f"{x}:maj")
        templates[i,:] = np.roll(mir_eval.chord.quality_to_bitmap('maj'),i)
    # minor chords
    for i,x in enumerate(pitch_class):
        chord_labels.append(f"{x}:min")
        templates[i+12,:] = np.roll(mir_eval.chord.quality_to_bitmap('min'),i)
    return chord_labels,templates

def plotTemplates(labels,templates):
    fig,ax = plt.subplots()
    ax.imshow(templates)
    ax.set_xticks([0,2,4,5,7,9,11])
    ax.set_xticklabels(["C","D","E","F","G","A","B"])
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels);
    return fig

def transcribeChromagram(t_chroma,chroma,transcriber="CRP"):
    if transcriber == "CRP":
        print("Transcribing with Chord Recognition Processor")
        crp = madmom.features.chords.DeepChromaChordRecognitionProcessor()
        prediction = crp(chroma)
        est_intervals = np.array([[x[0],x[1]] for x in prediction])
        est_labels = [x[2] for x in prediction]
        return est_intervals,est_labels
    elif transcriber == "HMM":
        print("Transcribing with Hidden Markov Model")
        model_path = "/home/max/ET-TI/Masterarbeit/models/hmm_model.pkl"   
        model = HMM.load_model(model_path)
        est_intervals,est_labels = model.predict(t_chroma,chroma)
        return est_intervals,est_labels
    elif transcriber == "TEMPLATE":
        print("Transcribing with Chord Templates")
        labels,templates = createChordTemplates()
        max_vals = np.max(chroma, axis=1)
        chroma_norm = chroma / (max_vals[:,None]+np.finfo(float).eps)
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
        return est_intervals,est_labels
    else:
        raise ValueError(f"Invalid Transcriber {transcriber}")
    

if __name__=='__main__':

    dataset = dataloader.MIRDataset("beatles",use_deep_chroma=True,align_chroma=True,split_nr=8)
   
    tracks = dataset.getTrackList()
    track_id = tracks.keys()

    for id in track_id:
        y,t_chroma,chroma,ref_intervals,ref_labels = dataset[id]
        results = {}
        for transcriber in ["CRP","TEMPLATE"]:
            # transcribe
            est_intervals,est_labels = transcribeChromagram(t_chroma,chroma,transcriber)
             # evaluate
            score,seg_score = evaluate.evaluateTranscription(est_intervals,est_labels,ref_intervals,ref_labels)
            ref_evalmatrix,est_evalmatrix = evaluate.createEvalMatrix(t_chroma,est_intervals,est_labels,ref_intervals,ref_labels)
            evaluate.plotEvaluationMatrix(ref_evalmatrix,est_evalmatrix)
            P, R, F, TP, FP, FN = evaluate.compute_eval_measures(ref_evalmatrix,est_evalmatrix)
            print(f"MajMin: {score}, MeanSeg: {seg_score}, P: {P}, R: {R}, F: {F}, TP: {TP}, FP: {FP}, FN: {FN}")