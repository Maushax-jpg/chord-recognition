import os
import mirdata
import json
import utils
from abc import ABC, abstractmethod

# list of tracks that should be excluded due to issues with the annotations / or lack of harmonic content
OUTLIERS = ['03_-_Anna_(Go_To_Him)', # tuning issues
            '10_-_Lovely_Rita', # tuning issues
            'CD1_-_05_-_Wild_Honey_Pie', # little harmonic content
            'CD1_-_06_-_The_Continuing_Story_of_Bungalow_Bill', # incoherent audiofile
            'CD2_-_12_-_Revolution_9',  # little harmonic content
            '02 Another One Bites The Dust', # little harmonic content
            '16 We Will Rock You', # little harmonic content
            "Stalker's Day Off (I've Been Hanging Around)", # faulty audio / issues with annotations
            'Stand Your Ground'  # faulty audio / issues with annotations
]

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
    def __init__(self,base_path):
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
    def __init__(self,base_path):
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
    def __init__(self,base_path):
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
    def __init__(self,base_path):
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
        "rwc_pop" : RWCPopDataset,
        "rw" : RWDataset,
        "queen" : QueenDataset
    }
    def __init__(self,dataset_name,base_path,source_separation):
        if dataset_name in self.DATASETS:
            self._dataset = self.DATASETS[dataset_name](base_path)
            if source_separation in ["none","drums","vocals","both"]:
                self._source_separation = source_separation
            else:
                raise ValueError(f"invalid source separation type: {source_separation}")
        else:
            raise NotImplementedError(f"Dataset {dataset_name} is not implemented!")
        
    def __getitem__(self,track_id):
        audiopath,annotationpath = self._dataset[track_id]

        if self._source_separation != "none": # modify audio_path
            basepath,filename = os.path.split(audiopath)
            filename = filename.rsplit('.', 1)[0]
            if self._source_separation == "drums":
                audiopath = os.path.join(basepath,f"{filename}_harmonic+vocals.mp3")
            elif self._source_separation == "vocals":
                audiopath = os.path.join(basepath,f"{filename}_harmonic+drums.mp3")
            elif self._source_separation == "both":
                audiopath = os.path.join(basepath,f"{filename}_harmonic.mp3")
        return audiopath,annotationpath

    def getTitle(self,track_id):
        return self._dataset._tracks[track_id].title
    
    def getTrackList(self):
        return self._dataset.getTrackList()
    
    def getExperimentSplits(self,split_nr):
        return self._dataset.getExperimentSplits(split_nr)
    


if __name__ == "__main__":

    import argparse
    import matplotlib.pyplot as plt
    import utils
    import numpy as np
    import mir_eval

    parser = argparse.ArgumentParser(
                    prog='dataloader DEMO',
                    description='iterates over available datasets and creates figures that describe the dataset')
    parser.add_argument('path',help='path to the mirdata basefolder')
    args = parser.parse_args()

    get_quality = lambda x : mir_eval.chord.split(x)[1]  # extract the root invariant chord quality


    chords = {}
    for dset in ["beatles","rwc_pop","rw","queen"]:
        dataset = Dataloader(dset,args.path,None)
        #chords = {}   # temporary dict for chords in a dataset
        for i in range(1,9):
            for track_id in dataset.getExperimentSplits(i):    
                audiopath,annotationpath = dataset[track_id]
                # y = utils.loadAudiofile(audiopath)
                intervals,labels = utils.loadChordAnnotations(annotationpath)
                
                # accumulate total chord duration
                for interval,label in zip(intervals,labels):
                    duration = chords.get(get_quality(label),0.0)
                    duration += (interval[1]-interval[0])
                    chords.update({get_quality(label):duration})
        chords.update({"N":chords.pop("")}) # rename No Chord label

    # compute total duration of audio in the dataset
    total_duration_hours = np.sum([x for x in chords.values()]) / 3600
    print(f"{dset} : {total_duration_hours:0.2f} hours of audio")

    # sort chord apperance in descending order 
    sorted_chords = sorted(chords.items(), key=lambda x:x[1],reverse=True)

    # extract the seven most frequent chord qualities
    nChords = 8
    labels = [x[0] for x in sorted_chords][:nChords] 
    labels.append("Others")
    durations = [x[1] for x in sorted_chords]
    durations_percent = durations / np.sum(durations) 
    durations_percent[nChords] = np.sum(durations_percent[nChords:]) # accumulate values at the end of the list
    durations_percent = durations_percent[:nChords+1] # crop list

    labels_percent = [f"{100*x:0.1f}%" for x in durations_percent]

    fig,ax = plt.subplots(figsize=(6,4))
    rects = ax.bar(labels,durations_percent)
    ax.bar_label(rects,labels_percent, label_type='edge')
    ax.set_ylim(0,0.8)
    ax.set_yticks([])
    ax.set_ylabel("Chord appearance in %")
    ax.set_xlabel("Root invariant chord quality")
    fig.tight_layout()
    plt.show()
