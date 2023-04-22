import os
import re
import pandas as pd
from tqdm import tqdm
import chromagram
from torch.utils.data import Dataset
import mir_eval

class BeatlesDataset(Dataset):
    def __init__(self, basepath, chroma_type='madmom'):
        self.filepaths_df = self.getFilePathDataframe(basepath)
        self.chroma_type = chroma_type

    def __len__(self):
        # returns rows of dataframe
        return self.filepaths_df.shape[0]

    def __getitem__(self, index):
        # return chroma and labels
        chroma = chromagram.getChroma(self.filepaths_df.loc[index,'audiopath'],self.chroma_type)
        annotations = mir_eval.io.load_labeled_intervals(self.filepaths_df.loc[index,'annotationpath'],'\t','#')
        return chroma,annotations
    
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
        chroma = chromagram.getChroma(self.filepaths_df.loc[index,'audiopath'],self.chroma_type)
        annotations = []
        return chroma,annotations
    
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