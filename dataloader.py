import os
import mirdata
import json
from abc import ABC, abstractmethod

class Dataset(ABC):
    """Abstract class for a dataset"""
    @abstractmethod
    def getFilePaths(self, track_id):
        """getter method for audiopath and chord annotationspath"""
        pass
    @abstractmethod
    def getExperimentSplits(self,split_nr):
        """returns a list of available Track_ID's for the given Split"""
        pass

    def __getitem__(self, track_id):
        track = self._tracks.get(track_id, None)
        if track is not None:
            return self.getFilePaths(track)
        else:
            raise IndexError
    
    def getTrackList(self):
        """returns a list of available Track_ID's for the dataset"""
        return list(self._tracks.keys())
    
    def loadJsonFile(self, file_path):
        try:
            with open(file_path, 'r', encoding='UTF-8') as file:
                return json.load(file)
        except FileNotFoundError:
            print(f"{file_path} file not found! Double-check the path.")
            raise FileNotFoundError

class RWDataset(Dataset):
    """Robbie Williams Dataset"""
    def __init__(self,base_path,hpss=None):
        self._hpss = hpss
        self._base_path = os.path.join(base_path,"robbiewilliams")
        self._tracks = {}
        data = self.loadJsonFile(os.path.join(self._base_path,"index.json"))
        for track in data["tracks"]:
            track_id,title,album,chords,keys,split = track.values()
            self._tracks[track_id] = customTrack(self._base_path,track_id,title,album,chords,keys,split)
        
    def getFilePaths(self,track):
        audiopath = track.audio_path
        annotationspath = track.chords_path
        return (audiopath,annotationspath)     
    
    def getExperimentSplits(self,split_nr):
        return [track_id for track_id,track in self._tracks.items() if track.split == split_nr]
    
class BeatlesDataset(Dataset):
    """Wrapper class for mirdata 'beatles' dataset"""
    def __init__(self,base_path,hpss=None):
        self._base_path = os.path.join(base_path,"beatles")
        self._mirdata_beatles =  mirdata.initialize("beatles", data_home=self._base_path)
        self._tracks = self._mirdata_beatles.load_tracks()

    def getFilePaths(self,track):
        audiopath = track.audio_path.replace("'","_") # files containing commas have been renamed
        annotationspath = track.chords_path.replace("'","_")
        return audiopath,annotationspath   
    
    def getExperimentSplits(self, split_nr):
        track_list = self.loadJsonFile(os.path.join(self._base_path,"splits",f"split_{split_nr}.json"))
        return list(track_list.keys())
    
   
class RWCPopDataset(Dataset):
    """Wrapper class for mirdata 'rwc_popular' dataset"""
    def __init__(self,base_path,hpss=None):
        self._base_path = os.path.join(base_path,"rwc_popular")
        self._mirdata_beatles =  mirdata.initialize("rwc_popular", data_home=self._base_path)
        self._tracks = self._mirdata_beatles.load_tracks()

    def getFilePaths(self,track):
        audiopath = track.audio_path.replace(".wav",".mp3")  # files are stored in mp3 format
        annotationspath = track.chords_path
        return audiopath,annotationspath     
    
    def getExperimentSplits(self, split_nr):
        track_list = self.loadJsonFile(os.path.join(self._base_path,"splits",f"split_{split_nr}.json"))
        return list(track_list.keys())
    
class QueenDataset(Dataset):
    """Wrapper class for mirdata 'queen' dataset"""
    def __init__(self,base_path,hpss=None):
        self._base_path = os.path.join(base_path,"queen")
        self._mirdata_queen =  mirdata.initialize("queen", data_home=self._base_path)
        self._tracks = self._mirdata_queen.load_tracks()
        self.removeTracks() # uncomment to use all available songs in this mirdata dataset 

    def getFilePaths(self,track):
        audiopath = track.audio_path
        annotationspath = track.chords_path
        return audiopath,annotationspath    
    
    def removeTracks(self):
        """function that removes Tracks without chord annotations from the dataset"""
        remove = []
        for key in self._tracks:
            if self._tracks[key].chords_path == None:
                remove.append(key)
        for key in remove:
            del self._tracks[key]
    
    def getExperimentSplits(self, split_nr):
        track_ids = []
        data = self.loadJsonFile(os.path.join(self._base_path,'index.json'))
        for track in data["tracks"]:
            track_id,_,_,split = track.values()
            if split == split_nr:
                track_ids.append(track_id) 
        return track_ids
    
class customTrack():
    """Class containing data for a track in a dataset (see mirdata.core.Track)"""
    def __init__(self,basepath,track_id,title,album,chords,keys,split):
        self.track_id = track_id
        self.title = title
        self.album = album
        self.audio_path = f"{basepath}/audio/{album}/{title}.mp3"
        self.chords_path = f"{basepath}/annotations/chords/{album}/{chords}"
        self.keys_path = f"{basepath}/annotations/keys/{album}/{keys}"
        self.split = split

    def __repr__(self):
        return f"Track(\naudio_path={self.audio_path}\nchords_path={self.chords_path}\nkeys_path={self.keys_path}\n)"

class Dataloader():
    DATASETS = {
        "beatles" : BeatlesDataset,
        "rwc_popular" : RWCPopDataset,
        "robbiewilliams" : RWDataset,
        "queen" : QueenDataset
    }
    def __init__(self,dataset_name,base_path):
        if dataset_name in self.DATASETS:
            self._dataset = self.DATASETS[dataset_name](base_path)
        else:
            print(f"Dataset {dataset_name} is not implemented!")
            raise NotImplementedError
        
    def __getitem__(self,track_id):
        return self._dataset[track_id]

    def getTitle(self,track_id):
        return self._dataset._tracks[track_id].title
    
    def getTrackList(self):
        return self._dataset.getTrackList()
    
    def getExperimentSplits(self,split_nr):
        return self._dataset.getExperimentSplits(split_nr)
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
                    prog='dataloader DEMO',
                    description='iterates over the dataset and prints out paths to audiofiles and ground truth annotations',
                    epilog='Text at the bottom of help')
    parser.add_argument('path',help='path to the mirdata basefolder')
    parser.add_argument('name',default = "queen", help='name of the dataset: "beatles","robbiewilliams","queen" or "rwc_popular"')
    args = parser.parse_args()
    
    dataset = Dataloader(args.name,args.path)

    for track_id in dataset.getTrackList():
        print(dataset.getTitle(track_id))
        audiopath,chords_path = dataset[track_id] 
        
        # ..
        # load files  etc.
        # .. 
