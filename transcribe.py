import numpy as np
import mir_eval
import utilities
import scipy.ndimage
import librosa

def harmonicPercussiveResidualSeparation(sig,beta=2,n_fft=1024,fs=22050):
    time_filter_length = 0.2 # seconds
    frequency_filter_length = 500 # Hz
    hs = n_fft // 2 # hopsize
    X = librosa.stft(sig,n_fft=n_fft,hop_length=hs,window='hann', center=True, pad_mode='constant')
    Y = np.abs(X) ** 2
    L_med_f = int(frequency_filter_length / (fs/n_fft))
    L_med_t = int(time_filter_length / (hs /fs))

    Y_p = scipy.ndimage.median_filter(Y, (L_med_f,1))
    Y_h = scipy.ndimage.median_filter(Y, (1,L_med_t))

    M_h = Y_h / (Y_p+np.finfo(float).eps) > beta
    M_p = Y_p / (Y_h+np.finfo(float).eps) >= beta
    M_r = np.logical_not(M_h + M_p)

    X_h = M_h * X
    X_p = M_p * X
    X_r = M_r * X

    y_harm = librosa.istft(X_h,n_fft=n_fft,hop_length=hs,window='hann', center=True, length=sig.size)
    y_perc = librosa.istft(X_p,n_fft=n_fft,hop_length=hs,window='hann', center=True, length=sig.size)
    y_res = librosa.istft(X_r,n_fft=n_fft,hop_length=hs,window='hann', center=True, length=sig.size)
    
    return y_harm,y_perc,y_res
    

def computeTemplateCorrelation(chroma,template_type="majmin"):
    templates,labels = utilities.createChordTemplates(template_type=template_type)
    correlation = np.matmul(templates.T,chroma)
    return correlation,labels

def transcribeWithTemplates(t_chroma,chroma,template_type="majmin"):
    correlation,labels = computeTemplateCorrelation(chroma,template_type)
    estimated_labels = [labels[np.argmax(correlation[:,i])] for i in range(chroma.shape[0])]
    estimated_intervals,estimated_labels =  utilities.createChordIntervals(t_chroma,estimated_labels)
    return estimated_intervals,estimated_labels

def applyPrefilter(chroma,filter_type="median",filterlength=17):
    return scipy.ndimage.median_filter(chroma, size=(1, args.prefilter_params))

def transcribeHMM(t_chroma,chroma,p=0.1,template_type="majmin"):
    correlation,labels = computeTemplateCorrelation(chroma,template_type)
    # neglect negative values of the correlation
    correlation = np.clip(correlation,0,100)
    A = uniform_transition_matrix(p,len(labels))
    B_O = correlation / (np.sum(correlation,axis=0)+np.finfo(float).tiny)
    C = np.ones((len(labels,))) * 1/len(labels)   # uniform initial state probability -> or start with "N"? 
    chord_HMM, _, _, _ = viterbi_log_likelihood(A, C, B_O)
    seq = np.argmax(chord_HMM,axis=0)
    labels_HMM = [labels[i] for i in seq]
    est_intervals,est_labels = utilities.createChordIntervals(t_chroma,labels_HMM)   
    return est_intervals,est_labels

def uniform_transition_matrix(p=0.01, N=24):
    """Computes uniform transition matrix
    source: https://www.audiolabs-erlangen.de/resources/MIR/FMP/C5/C5S3_ChordRec_HMM.html
    Notebook: C5/C5S3_ChordRec_HMM.ipynb

    Args:
        p (float): Self transition probability (Default value = 0.01)
        N (int): Column and row dimension (Default value = 24)

    Returns:
        A (np.ndarray): Output transition matrix
    """
    off_diag_entries = (1-p) / (N-1)     # rows should sum up to 1
    A = off_diag_entries * np.ones([N, N])
    np.fill_diagonal(A, p)
    return A


def viterbi_log_likelihood(A, C, B_O):
    """Viterbi algorithm (log variant) for solving the uncovering problem
    source: https://www.audiolabs-erlangen.de/resources/MIR/FMP/C5/C5S3_ChordRec_HMM.html
    Notebook: C5/C5S3_Viterbi.ipynb

    Args:
        A (np.ndarray): State transition probability matrix of dimension I x I
        C (np.ndarray): Initial state distribution  of dimension I
        B_O (np.ndarray): Likelihood matrix of dimension I x N

    Returns:
        S_opt (np.ndarray): Optimal state sequence of length N
        S_mat (np.ndarray): Binary matrix representation of optimal state sequence
        D_log (np.ndarray): Accumulated log probability matrix
        E (np.ndarray): Backtracking matrix
    """
    I = A.shape[0]    # Number of states
    N = B_O.shape[1]  # Length of observation sequence
    tiny = np.finfo(float).tiny
    A_log = np.log(A + tiny)
    C_log = np.log(C + tiny)
    B_O_log = np.log(B_O + tiny)

    # Initialize D and E matrices
    D_log = np.zeros((I, N))
    E = np.zeros((I, N-1)).astype(np.int32)
    D_log[:, 0] = C_log + B_O_log[:, 0]

    # Compute D and E in a nested loop
    for n in range(1, N):
        for i in range(I):
            temp_sum = A_log[:, i] + D_log[:, n-1]
            D_log[i, n] = np.max(temp_sum) + B_O_log[i, n]
            E[i, n-1] = np.argmax(temp_sum)

    # Backtracking
    S_opt = np.zeros(N).astype(np.int32)
    S_opt[-1] = np.argmax(D_log[:, -1])
    for n in range(N-2, -1, -1):
        S_opt[n] = E[int(S_opt[n+1]), n]

    # Matrix representation of result
    S_mat = np.zeros((I, N)).astype(np.int32)
    for n in range(N):
        S_mat[S_opt[n], n] = 1

    return S_mat, S_opt, D_log, E

def evaluateTranscription(est_intervals,est_labels,ref_intervals,ref_labels,scheme="majmin"):
    """evaluate Transcription score according to MIREX scheme"""
    est_intervals, est_labels = mir_eval.util.adjust_intervals(
            est_intervals, est_labels, ref_intervals.min(),
            ref_intervals.max(), mir_eval.chord.NO_CHORD,
            mir_eval.chord.NO_CHORD)
    (intervals,ref_labels,est_labels) = mir_eval.util.merge_labeled_intervals(
        ref_intervals, ref_labels, est_intervals, est_labels)
    durations = mir_eval.util.intervals_to_durations(intervals)
    if scheme == "majmin":
        comparisons = mir_eval.chord.majmin(ref_labels, est_labels)
    elif scheme == "triads":
        comparisons = mir_eval.chord.triads(ref_labels, est_labels)
    elif scheme == "tetrads":
        comparisons = mir_eval.chord.tetrads(ref_labels, est_labels)
    elif scheme == "majmin_sevenths":
        comparisons = mir_eval.chord.tetrads(ref_labels, est_labels)
    score = round(mir_eval.chord.weighted_accuracy(comparisons, durations),2)
    mean_seg_score = round(mir_eval.chord.seg(ref_intervals, est_intervals),2)
    return score,mean_seg_score

def transcribeChromagram(t_chroma,chroma,**kwargs):
    ## prefilter ## 
    if kwargs.get("prefilter",None) == "median":
        N = kwargs.get("prefilter_length",15)
        chroma = scipy.ndimage.median_filter(chroma, size=(1, N))
    ## pattern matching ##         
    vocabulary = kwargs.get("vocabulary","majmin")
    templates,labels = utilities.createChordTemplates(template_type=vocabulary) 
    correlation = np.matmul(templates.T,chroma)
    correlation = np.clip(correlation,a_min=0,a_max=1) # disregard negative values
    ## postfilter ##
    postfilter = kwargs.get("postfilter",None)
    if postfilter is None:
        unfiltered_labels = [labels[np.argmax(correlation[:,i])] for i in range(chroma.shape[1])]
        est_intervals,est_labels =  utilities.createChordIntervals(t_chroma,unfiltered_labels)
    elif postfilter == "median":
        N = kwargs.get("postfilter_length",15)
        correlation = scipy.ndimage.median_filter(correlation, size=(1, N))
        unfiltered_labels = [labels[np.argmax(correlation[:,i])] for i in range(chroma.shape[1])]
        est_intervals,est_labels =  utilities.createChordIntervals(t_chroma,unfiltered_labels)
    elif postfilter == "hmm":
        p = kwargs.get("transition_prob",0.1)
        est_intervals,est_labels = transcribeHMM(t_chroma,chroma,p=p,template_type=vocabulary)
    return est_intervals, est_labels


if __name__ == "__main__":
    import argparse
    import features
    import transcribe
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path',help='filepath to song')
    parser.add_argument('--hps', help='choose harmonic percussive source separation type: hprs, htdemucs')
    parser.add_argument('--prefilter', help='choose prefilter type: median, none')
    parser.add_argument('--prefilter_params', help='choose prefilter length: n',type=int,default=15)
    parser.add_argument('--chroma_type', help='choose chromagram type: cqt, crp, dcp')
    parser.add_argument('--vocabulary', help='choose chord vocabulary: majmin, triads, triads_extended,majmin_sevenths',default='majmin')
    parser.add_argument('--postfilter',help='choose postfilter type: median, hmm',default=None)
    parser.add_argument('--postfilter_params', help='choose postfilter length: n',type=int,default=1)
    parser.add_argument('--output_path', help='specify output path for transcription results',default=None)
    parser.add_argument('--filename', help='specify output filename for transcription results',default="transcription.chords")
    args = parser.parse_args()
    
    audio_path = args.file_path
    if audio_path is None:
        print("Please specify input path for audiofile!")
        quit()
    time_vector,signal = utilities.loadAudio(audio_path)

    ## HPS ##
    # not implemented yet

    ## compute chromagram ##
    if args.chroma_type == "crp":
        t_chroma, chroma = features.crpChroma(signal)
    elif args.chroma_type == "dcp":
        t_chroma, chroma = features.deepChroma(signal)
    elif args.chroma_type == "cqt":
        t_chroma, chroma = features.cqtChroma(signal)
    else:
        print(f"invalid chroma type {args.chroma_type}")
        quit()

    ## prefilter ## 
    if args.prefilter == "median":
        chroma = scipy.ndimage.median_filter(chroma, size=(1, args.prefilter_params))

    ## pattern matching ## 
    templates,labels = utilities.createChordTemplates(template_type=args.vocabulary) 
    correlation = np.matmul(templates.T,chroma)
    correlation = np.clip(correlation,a_min=0,a_max=1) # disregard negative values

    ## postfilter ##
    if args.postfilter is None:
        unfiltered_labels = [labels[np.argmax(correlation[:,i])] for i in range(chroma.shape[1])]
        est_intervals,est_labels =  utilities.createChordIntervals(t_chroma,unfiltered_labels)
    elif args.postfilter == "median":
        correlation = scipy.ndimage.median_filter(correlation, size=(1, args.postfilter_params))
        unfiltered_labels = [labels[np.argmax(correlation[:,i])] for i in range(chroma.shape[1])]
        est_intervals,est_labels =  utilities.createChordIntervals(t_chroma,unfiltered_labels)
    elif args.postfilter == "hmm":
        est_intervals,est_labels = transcribe.transcribeHMM(t_chroma,chroma,p=0.1,template_type=args.vocabulary)

    if args.output_path is not None:
        fpath = f"{args.output_path}/{args.filename}"
        f = open(fpath, "w")
        # save to a .chords file
        for interval,label in zip(est_intervals,est_labels):
            f.write(f"{interval[0]:0.6f}\t{interval[1]:0.6f}\t{label}\n")
        f.close()

        fig,ax = plt.subplots(3,2,height_ratios=(1,3,10),width_ratios=(9.5,.5))
        utilities.plotChordAnnotations(ax[0,0],(est_intervals,est_labels),(0,10))
        ax[1,0].plot(time_vector,signal )
        ax[1,0].set_ylim(0,1)
        ax[1,0].set_xlim(0,10)
        ax[1,1].set_axis_off()
        ax[0,1].set_axis_off()
        img = utilities.plotChromagram(ax[2,0],t_chroma,chroma,None,None,vmin=-np.max(chroma),vmax=np.max(chroma),cmap='bwr')
        fig.colorbar(img,cax=ax[2,1],cmap="bwr")
        ax[2,0].set_xlim([0,10])
        plt.savefig(f"{args.output_path}/result_preview.png")
        
    intervals,  labels = utilities.loadAnnotations(f"{args.output_path}/{args.filename}")
    print(labels)
