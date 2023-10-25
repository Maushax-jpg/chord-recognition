import numpy as np
import utilities

def computeTemplateCorrelation(chroma,template_type="majmin"):
    templates,labels = utilities.createChordTemplates(template_type=template_type)
    correlation = np.dot(templates,chroma.T)
    return correlation,labels

def transcribeWithTemplates(t_chroma,chroma,template_type="majmin"):
    correlation,labels = computeTemplateCorrelation(chroma,template_type)
    estimated_labels = [labels[np.argmax(correlation[:,i])] for i in range(chroma.shape[0])]
    estimated_intervals,estimated_labels =  utilities.createChordIntervals(t_chroma,estimated_labels)
    return estimated_intervals,estimated_labels