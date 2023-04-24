import os
import re
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset
import mir_eval
import madmom
import librosa
import librosa.display
import numpy as np
import scipy
import audioread.ffdec

SAMPLING_RATE = 44100
HOP_LENGTH = 4410

class BeatlesDataset(Dataset):
    def __init__(self, basepath=None, chroma_type='madmom',beat_align=True):
        self.filepaths_df = self.getFilePathDataframe(basepath)
        self.chroma_type = chroma_type
        self.beat_align = beat_align
        self.chroma_processor = madmom.audio.chroma.DeepChromaProcessor()
        self.activations_processor = madmom.features.beats.RNNBeatProcessor()
        self.beat_processor = madmom.features.beats.BeatTrackingProcessor(fps=100)

    def __len__(self):
        """returns rows of dataframe"""
        return self.filepaths_df.shape[0]

    def __getitem__(self, index):
        """explain away"""
        t_chroma,chroma = self.getChroma(self.filepaths_df.loc[index,'audiopath'],self.chroma_type)
        ref_intervals,ref_labels = mir_eval.io.load_labeled_intervals(self.filepaths_df.loc[index,'annotationpath'],'\t','#')
        return t_chroma,chroma,ref_intervals,ref_labels
    
    def getChroma(self,filepath,chroma_type='librosa'):
        if chroma_type == 'librosa':
            with audioread.ffdec.FFmpegAudioFile(filepath) as file:
                y, sr = librosa.load(file,sr=SAMPLING_RATE)
                # harmonic percussive sound separation
            y_harm = librosa.effects.harmonic(y=y, margin=8)
            # calculate chroma
            chroma_harm = librosa.feature.chroma_cqt(y=y_harm, sr=SAMPLING_RATE,hop_length=HOP_LENGTH)
            # use non-local filtering
            chroma_filter = np.minimum(chroma_harm,
                            librosa.decompose.nn_filter(chroma_harm,
                                                        aggregate=np.median,
                                                        metric='cosine'))
            #  apply horizontal median filter
            chroma = scipy.ndimage.median_filter(chroma_filter, size=(1, 9)).T
            t = np.linspace(0,chroma.shape[0]*HOP_LENGTH/SAMPLING_RATE,chroma.shape[0])
        elif chroma_type == 'madmom':
            chroma = self.chroma_processor(filepath)
            t = np.linspace(0,chroma.shape[0]*HOP_LENGTH/SAMPLING_RATE,chroma.shape[0])
        if self.beat_align:
            beat_activations = self.activations_processor(filepath)
            beats = self.beat_processor(beat_activations)
            chroma = self.alignChroma(t,chroma,beats)
        return t,chroma

    def alignChroma(self,t_chroma,chroma,beats):
        """beat align chroma"""
        chroma_aligned = chroma
        
        for b0,b1 in zip(beats[:-1],beats[1:]):
            try:
                idx0 = np.argwhere(t_chroma >= b0)[0][0]
                idx1 = np.argwhere(t_chroma >= b1)[0][0]
            except IndexError:
                idx1 = t_chroma.shape[0]
            # use non-local filtering
            chroma_filter = np.minimum(chroma[idx0:idx1,:].T,
                            librosa.decompose.nn_filter(chroma[idx0:idx1,:].T,
                                                        aggregate=np.median,
                                                        metric='cosine'))
            # apply horizontal median filter
            chroma_aligned[idx0:idx1,:] = scipy.ndimage.median_filter(chroma_filter, size=(1, 9)).T
        return chroma_aligned
    
    def getFilepaths(self,index):
        try:
            audio = self.filepaths_df.loc[index,'audiopath']
            annotations = self.filepaths_df.loc[index,'annotationpath']
            return audio,annotations        
        except KeyError:
            print(f"KeyError: Index not found in dataset!")
            return None
    
    def getTitle(self,index):
        try:
            return self.filepaths_df.loc[index,'title']
        except KeyError:
            print(f"KeyError: Index not found in dataset!")
            return None
    
    def getFilePathDataframe(self,basepath):
        '''This function returns a list of tuples containing songname and paths to .mp3 and .chords files
            basepath_labels: path to chord annotations
            basepath_audio: path to beatles albums'''
        # this code snippet searches for a matching annotation file in basepath_labels for every song found in
        # basepath_audio. The filenames differ in syntax so regular expressions are used
        if not basepath:
            return None
        basepath_audio = os.path.join(basepath,"audio")
        basepath_labels = os.path.join(basepath,"annotations/chords")
        labelfiles= os.listdir(basepath_labels)
        songs = []
        for folder in os.listdir(basepath_audio):
            # construct filename
            filename = re.sub(r"_-_", "_", folder)
            filename = "beatles_"+filename+"_"
            folderpath = os.path.join(basepath_audio,folder)
            for file in  os.listdir(folderpath):
                temp = re.sub(r"_-_", "_", file)
                audiofile = filename+temp
                temp = re.sub(r"\.mp3\b","",audiofile)
                title = re.sub(r"[\'\.\!\,]", "", temp)
                # search correct file with labels for the song
                found= False
                for labelfile in labelfiles:
                    if labelfile.lower().startswith(title.lower()):
                        audiopath = os.path.join(folderpath,file)
                        labelpath = os.path.join(basepath_labels,labelfile)
                        songs.append((title,audiopath,labelpath)) 
                        found = True
                        break
                if not found:
                    raise ValueError(f"songname not found: {title}")
            
        return pd.DataFrame(songs,columns=['title','audiopath','annotationpath'])

class ChordSequencesDataset(Dataset):
    def __init__(self, basepath, chroma_type='madmom'):
        self.filepaths_df = self.getFilePathDataframe(basepath)
        self.chroma_type = chroma_type
        
    def __getitem__(self, index):
        # return chroma and labels
        #chroma = chromagram.getChroma(self.filepaths_df.loc[index,'audiopath'],self.chroma_type)
        annotations = []
        #return chroma,annotations
    
    def getFilepaths(self,index):
        try:
            audio = self.filepaths_df.loc[index,'audiopath']
            annotations = self.filepaths_df.loc[index,'annotationpath']
            return audio,annotations
        except KeyError:
            print(f"KeyError: Index not found in dataset!")
            return None
        
    def getTitle(self,index):
        try:
            return self.filepaths_df.loc[index,'title']
        except KeyError:
            print(f"KeyError: Index not found in dataset!")
            return None
        
    def getFilePathDataframe(self,basepath):
        """ 
        This function returns a Pandas Dataframe with filepaths to all available audio and annotation files 
        for the Chord_sequences Dataset (https://www.idmt.fraunhofer.de/en/publications/datasets/chord-sequences.html)  
        @basepath: path to the dataset
        """
        filepath_df = pd.DataFrame(columns=['title','audiopath','annotationpath'])
        json_paths = os.listdir(basepath)
        json_paths.sort()
        pattern = re.compile(r"\d+_")
        result = pattern.search(json_paths[0])
        id = result.group()[:-1]
        id_list = [id]
        for file in json_paths[1:]:
            result = pattern.search(file)
            temp = result.group()[:-1]
            if id != temp:
                id_list.append(temp)
                id=temp

        for id in id_list:
            col = {}
            col["title"] = id+"_anchor"
            col["annotationpath"] = os.path.join(basepath,id+"_anchor.json")
            #col["mid"] = os.path.join(basepath,id+"_anchor.mid")
            col["audiopath"] = os.path.join(basepath,id+"_anchor.wav")
            filepath_df.loc[len(filepath_df),:] = col
            col = {}
            col["title"] = id+"_negative"
            col["annotationpath"] = os.path.join(basepath,id+"_negative.json")
            #col["mid"] = os.path.join(basepath,id+"_anchor.mid")
            col["audiopath"] = os.path.join(basepath,id+"_negative.wav")
            filepath_df.loc[len(filepath_df),:] = col        
            col = {}
            col["title"] = id+"_positive"
            col["annotationpath"] = os.path.join(basepath,id+"_positive.json")
            #col["mid"] = os.path.join(basepath,id+"_anchor.mid")
            col["audiopath"] = os.path.join(basepath,id+"_positive.wav")
            filepath_df.loc[len(filepath_df),:] = col
        return filepath_df
