import os
from torch.utils.data import Dataset
import mir_eval
import mirdata
import json

SAMPLING_RATE = 44100
HOP_LENGTH = 4410

class MIRDataset(Dataset):
    """
    Dataloader class for MIR Datasets. Available Datasets are "guitarset","beatles" and "queen"
    """
    def __init__(self,name,basepath="/home/max/ET-TI/Masterarbeit/mirdata/",split_nr=0):
        super().__init__()
        self._path = basepath
        self._split_nr = split_nr
        self._dataset = self.loadDataset(name)
        self._tracks = self._dataset.load_tracks()

    def __len__(self):
        """returns rows of dataframe"""
        return len(self._dataset.track_ids)
    
    def __getitem__(self, track_id):
        target = self.getAnnotations(track_id)
        audio_path = self._tracks[track_id].audio_path
        return audio_path, target
    
    def getTrackIDs(self):
        """returns list of available track IDs that can be used to access the dataset tracks"""
        return self._dataset.track_ids
    
    def getTitle(self,track_id):
        return self._tracks[track_id].title
    
    def getTrackList(self):
        """returns list of available track names"""
        with open(f"/home/max/ET-TI/Masterarbeit/mirdata/beatles/splits/split_{self._split_nr}.json", 'r', encoding='UTF-8') as file:
            track_list = json.load(file)
        return track_list

    def loadDataset(self,name):
        if name != "beatles":
            raise ValueError(f"Dataset {name} not available!")
        return mirdata.initialize(name, data_home=os.path.join(self._path,name))
    
    def getAnnotations(self,track_id):
        ref_intervals,ref_labels = mir_eval.io.load_labeled_intervals(self._tracks[track_id].chords_path,' ','#')
        return ref_intervals,ref_labels
