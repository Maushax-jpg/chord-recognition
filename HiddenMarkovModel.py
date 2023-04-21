import hmmlearn 

class HiddenMarkovModel(hmmlearn.hmm):
    def __init__(self, path):
        self.states = {}

    def calculateMean(self,features,target):
        """calculate mean value of labeled featurevector for all states"""
        