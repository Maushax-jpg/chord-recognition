import numpy as np
import utilities

def computeTemplateCorrelation(chroma,template_type="majmin"):
    labels,templates = utilities.createChordTemplates(templateType=template_type)
    max_vals = np.max(chroma, axis=1)
    chroma_norm = chroma / (max_vals[:,None]+np.finfo(float).eps)
    correlation = np.dot(templates,chroma_norm.T)
    return correlation,labels

def transcribeWithTemplates(t_chroma,chroma,template_type="majmin"):
    correlation,labels = computeTemplateCorrelation(chroma,template_type)
    estimated_labels = [labels[np.argmax(correlation[:,i])] for i in range(chroma.shape[0])]
    estimated_intervals,estimated_labels =  utilities.createChordIntervals(t_chroma,estimated_labels)
    return estimated_intervals,estimated_labels