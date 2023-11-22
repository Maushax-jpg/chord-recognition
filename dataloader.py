import os
import mirdata
import json
from abc import ABC, abstractmethod

class Dataset(ABC):
    """Abstract class for a dataset"""
    @abstractmethod
    def __getitem__(self, track_id):
        """returns the audiopaths and chordannotations for the given Track_ID"""
        pass

    @abstractmethod
    def getTrackList(self):
        """returns a list of available Track_ID's for the dataset"""
        pass
    @abstractmethod
    def getExperimentSplits(self,split_nr):
        """returns a list of available Track_ID's for the given Split"""
        pass

class RWDataset(Dataset):
    """Robbie Williams Dataset"""
    def __init__(self,base_path,hpss=None):
        self._hpss = hpss
        self._base_path = os.path.join(base_path,"robbiewilliams")
        self._tracks = {}
        try:
            with open(os.path.join(self._base_path,'index.json')) as json_file:
                data = json.load(json_file)
                for track in data["tracks"]:
                    track_id,title,album,chords,keys,split = track.values()
                    self._tracks[track_id] = customTrack(self._base_path,track_id,title,album,chords,keys,split)
        except FileNotFoundError:
            print(f"index.json file not found! Double check path")
            raise FileNotFoundError
        
    def __getitem__(self,track_id):
        track = self._tracks.get(track_id,None)
        audiopath = track.audio_path
        annotationspath = track.chords_path
        return (audiopath,annotationspath)     
    
    def getTrackList(self):
        return self._tracks.keys()
    
    def getExperimentSplits(self,split_nr):
        return [track_id for track_id,track in self._tracks.items() if track.split == split_nr]
    
class BeatlesDataset(Dataset):
    """Wrapper class for mirdata 'beatles' dataset"""
    def __init__(self,base_path,hpss=None):
        self._base_path = os.path.join(base_path,"beatles")
        self._mirdata_beatles =  mirdata.initialize("beatles", data_home=self._base_path)
        self._tracks = self._mirdata_beatles.load_tracks()

    def __getitem__(self,track_id):
        track = self._tracks.get(track_id,None)
        audiopath = track.audio_path
        annotationspath = track.chords_path
        return (audiopath,annotationspath)     
    
    def getTrackList(self):
        return self._tracks.keys()
    
    def getExperimentSplits(self, split_nr):
        try:
            path = os.path.join(self._base_path,"splits",f"split_{split_nr}.json")
            with open(path, 'r', encoding='UTF-8') as file:
                track_list = json.load(file)
        except FileNotFoundError:
            print(f"split_{split_nr}.json file not found! Double check path")
            quit()
        return list(track_list.keys())
    
   
class RWCPopDataset(Dataset):
    """Wrapper class for mirdata 'rwc_popular' dataset"""
    def __init__(self,base_path,hpss=None):
        self._base_path = os.path.join(base_path,"rwc_popular")
        self._mirdata_beatles =  mirdata.initialize("rwc_popular", data_home=self._base_path)
        self._tracks = self._mirdata_beatles.load_tracks()

    def __getitem__(self,track_id):
        track = self._tracks.get(track_id,None)
        audiopath = track.audio_path.replace(".wav",".mp3")  # files are stored in mp3 format
        annotationspath = track.chords_path
        return (audiopath,annotationspath)     
    
    def getTrackList(self):
        return self._tracks.keys()
    
    def getExperimentSplits(self, split_nr):
        try:
            path = os.path.join(self._base_path,"splits",f"split_{split_nr}.json")
            with open(path, 'r', encoding='UTF-8') as file:
                track_list = json.load(file)
        except FileNotFoundError:
            print(f"split_{split_nr}.json file not found! Double check path")
            quit()
        return list(track_list.keys())
    
class QueenDataset(Dataset):
    """Wrapper class for mirdata 'queen' dataset"""
    def __init__(self,base_path,hpss=None):
        self._base_path = os.path.join(base_path,"queen")
        self._mirdata_queen =  mirdata.initialize("queen", data_home=self._base_path)
        self._tracks = self._mirdata_queen.load_tracks()
        self.removeTracks() # uncomment to use all available songs in this mirdata dataset 

    def __getitem__(self,track_id):
        track = self._tracks.get(track_id,None)
        audiopath = track.audio_path
        annotationspath = track.chords_path
        return (audiopath,annotationspath)     
    
    def removeTracks(self):
        """function that removes Tracks without chord annotations from the dataset"""
        remove = []
        for key in self._tracks:
            if self._tracks[key].chords_path == None:
                remove.append(key)
        for key in remove:
            del self._tracks[key]

    def getTrackList(self):
        return self._tracks.keys()
    
    def getExperimentSplits(self, split_nr):
        track_ids = []
        try:
            with open(os.path.join(self._base_path,'index.json')) as json_file:
                data = json.load(json_file)
                for track in data["tracks"]:
                    print(track,split_nr)
                    track_id,title,album,split = track.values()
                    print(type(split),type(split_nr))
                    if split == split_nr:
                        track_ids.append(track_id) 
                return track_ids
        except FileNotFoundError:
            print(f"index.json file not found! Double check path")
            raise FileNotFoundError

    
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

if __name__ == "__main__":
    ### DEMO ### 
    PATH = "/home/max/ET-TI/Masterarbeit/mirdata/" # adjust path to data
    dataset = BeatlesDataset(PATH)
    # dataset = QueenDataset(PATH)
    # dataset = RWCPopDataset(PATH)
    # dataset = RWDataset(PATH)

    track_ids = dataset.getTrackList()
    for track_id in track_ids:  # iterate over all available tracks
        audiopath,annotations = dataset[track_id] 
        print(audiopath)
        # .. 
        # load audio, load annotations
        # .. 
