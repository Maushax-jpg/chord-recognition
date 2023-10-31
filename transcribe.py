import numpy as np
import utilities

def computeTemplateCorrelation(chroma,template_type="majmin"):
    templates,labels = utilities.createChordTemplates(template_type=template_type)
    correlation = np.matmul(templates.T,chroma)
    return correlation,labels

def transcribeWithTemplates(t_chroma,chroma,template_type="majmin"):
    correlation,labels = computeTemplateCorrelation(chroma,template_type)
    estimated_labels = [labels[np.argmax(correlation[:,i])] for i in range(chroma.shape[0])]
    estimated_intervals,estimated_labels =  utilities.createChordIntervals(t_chroma,estimated_labels)
    return estimated_intervals,estimated_labels

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