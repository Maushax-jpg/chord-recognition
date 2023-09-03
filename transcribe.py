import circularPitchSpace as cps
import madmom
import dataloader
import evaluate
import mir_eval
import numpy as np

def postprocessing(time_vector,unfiltered_labels,complexity,complexity_threshold=4.0,minimum_duration=0.4):
    intervals = []
    labels = []
    t_start = time_vector[0]
    current_label = "N"
    for i,label in enumerate(unfiltered_labels[:-1]):
        if label == current_label:
            continue
        elif complexity[i] > complexity_threshold:
            continue
        
        # check duration of chord change (excluding first chord change!)
        if time_vector[i] - t_start < minimum_duration and len(intervals) > 0:
            # interval too short, extend previous interval  and discard chord change
            intervals[-1][1] = time_vector[i]
        else:
            labels.append(current_label)
            intervals.append([t_start,time_vector[i]])
        # initialize new interval
        current_label = label
        t_start = time_vector[i]
    intervals = np.array(intervals)
    return intervals,labels


def transcribeTemplates(time_vector,chroma,complexity,complexity_threshold=4,minimum_duration=0.5):
    labels,templates = createChordTemplates()
    max_vals = np.max(chroma, axis=1)
    chroma_norm = chroma / (max_vals[:,None]+np.finfo(float).eps)
    correlation = np.dot(templates,chroma_norm.T)
    unfiltered_labels = [labels[np.argmax(correlation[:,i])] for i in range(chroma.shape[0])]

    intervals,labels = postprocessing(time_vector,unfiltered_labels,complexity,complexity_threshold,minimum_duration)
    return intervals,labels

def transcribeCPS(time_vector,chroma,complexity,complexity_threshold=4.0,minimum_duration=0.4):
    r_F,r_FR,r_TR,r_DR = cps.transformChroma(chroma)

    # create normalized major key prototypes
    prototypes = np.zeros((12,7,12),dtype=float)#
    chord_qualities = ["maj","min","min","maj","maj","min","dim"]
    pitch_classes = [0,2,4,5,7,9,11]
    templates =  np.array([np.roll(mir_eval.chord.quality_to_bitmap(quality)/3,pitch) for quality,pitch in zip(chord_qualities,pitch_classes)])
    for i in range(12):
        prototypes[i,:,:] = np.roll(templates,i,axis=1) # normalize with l1 norm

    # estimate key for analysis
    keys = np.argmax(cps.getPitchClassEnergyProfile(chroma),axis=1)

    unfiltered_labels = []
    for t in range(chroma.shape[0]):
        pitch_class_index = keys[t]
        __build_class__,x_FR,x_TR,x_DR = cps.transformChroma(prototypes[pitch_class_index,:,:])
        # calculate minimum distance between prototype chord and feature
        d_TR = np.abs(r_TR[t,pitch_class_index] - x_TR[:,pitch_class_index])
        d_FR = np.abs(r_FR[t,pitch_class_index] - x_FR[:,pitch_class_index])
        # d_DR = np.abs(r_DR[t,pitch_class_index] - x_DR[:,pitch_class_index])
        index_min_distance = np.argsort(d_TR+d_FR)[0]
        estimated_pitch_class = (pitch_class_index + pitch_classes[index_min_distance]) % 12
        unfiltered_labels.append(cps.pitch_classes[estimated_pitch_class].name)
    intervals,labels = postprocessing(time_vector,unfiltered_labels,complexity,
                                      complexity_threshold=complexity_threshold,minimum_duration=minimum_duration)
    return intervals,labels


def transcribeChromagram(t_chroma,chroma,transcriber="CRP",entropy=None):
    if transcriber == "CRP":
        print("Transcribing with Chord Recognition Processor")
        crp = madmom.features.chords.DeepChromaChordRecognitionProcessor()
        prediction = crp(chroma)
        est_intervals = np.array([[x[0],x[1]] for x in prediction])
        est_labels = [x[2] for x in prediction]
        return est_intervals,est_labels
    elif transcriber == "CPS":
        print("Transcribing with Circular Pitch space")        
        est_intervals,est_labels = cps.transcribeChromagram(t_chroma,chroma)
        intervals = []
        labels =[]
        skip = []
        for i,x in enumerate(zip(est_intervals,est_labels)):
            interval,label = x
            try:
                mean_entropy = np.mean(entropy[int(10*interval[0]):int(10*interval[1])])
                if mean_entropy < 1.2:
                    skip.append(i)
                    est_intervals[i-1][1] = est_intervals[i+1][0]  # adjust interval bounds 
            except IndexError:
                continue
            
        for i in range(len(est_labels)):
            if i in skip:
                continue
            else:
                labels.append(est_labels[i])
                intervals.append(est_intervals[i])
        intervals = np.array(intervals)
        return intervals,labels
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

    dataset = dataloader.MIRDataset("beatles",use_deep_chroma=True,align_chroma=True,split_nr=3)
   
    tracks = dataset.getTrackList()
    track_id = tracks.keys()

    for id in track_id:
        audio,features,target = dataset[id]
        ref_intervals,ref_labels = target
        chroma = features["chroma"]
        t_chroma = features["time"]
        rms = features["rms"]
        entropy = features["entropy"]
        title = dataset.getTitle(id)
        results = {}
        for transcriber in ["CPS","TEMPLATE"]:
            # transcribe
            est_intervals,est_labels = transcribeChromagram(t_chroma,chroma,transcriber,entropy)

             # evaluate
            score,seg_score = evaluate.evaluateTranscription(est_intervals,est_labels,ref_intervals,ref_labels)
            ref_evalmatrix,est_evalmatrix = evaluate.createEvalMatrix(t_chroma,est_intervals,est_labels,ref_intervals,ref_labels)
            evaluate.plotEvaluationMatrix(ref_evalmatrix,est_evalmatrix,title,transcriber)
            P, R, F, TP, FP, FN = evaluate.compute_eval_measures(ref_evalmatrix,est_evalmatrix)
            print(f"MajMin: {score}, MeanSeg: {seg_score}, P: {P}, R: {R}, F: {F}, TP: {TP}, FP: {FP}, FN: {FN}")