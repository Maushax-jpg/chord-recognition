import os
import re
import pandas as pd
import numpy as np
from tqdm import tqdm
import chromagram
from torch.utils.data import Dataset
import mir_eval

class BeatlesDataset(Dataset):
    def __init__(self,basepath="/home/max/ET-TI/Masterarbeit/prototyping/data/beatles/",chroma_type='madmom'):
        self.filepaths_df = self.getPaths(basepath)
        self.chroma_df = pd.DataFrame()
        self.annotations_df = pd.DataFrame()
        self.chroma_type = chroma_type

    def __getitem__(self, index):
        # return chroma and labels
        chroma = chromagram.getChroma(self.filepaths_df.loc[index,'audiopath'],self.chroma_type)
        annotations = mir_eval.io.load_labeled_intervals(self.filepaths_df.loc[index,'annotationpath'],'\t','#')
        title = self.filepaths_df.loc[index,'title']
        return title,chroma,annotations

    def __len__(self):
        return len(self.filepaths_df)

    def getPaths(self,basepath):
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
    def __init__(self, path):
        self.chroma_df = pd.DataFrame()
        self.annotations_df = pd.DataFrame()

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass

chord_labels_majmin = {'N':0,'C':1,'C#':2,'D':3,'D#':4,'E':5,'F':6,'F#':7,'G':8,'G#':9,'A':10,'A#':11,'B':12,
    'C:min':13,'C#:min':14,'D:min':15,'D#:min':16,'E:min':17,'F:min':18,'F#:min':19,'G:min':20,'G#:min':21,
    'A:min':22,'A#:min':23,'B:min':24
}
enharmonic_notes = {'Db':'C#','Eb':'D#','Gb':'F#','Ab':'G#','Bb':'A#'} 



def getBeatlesAnnotations(label_path):
    labels_df = pd.read_csv(label_path,sep='\t',names=["tstart","tend","label"])
    return labels_df

def getBeatlesChroma(songs,chroma_type='librosa'):
    """creates a dataframe for a given list of beatles songs
       a songlist contaings: songname,audiopath,labelpath
    """
    # create dataframe
    chroma_labels = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
    labels = ["label"]
    labels.extend(chroma_labels)
    chroma_df = pd.DataFrame(columns=labels)

    for _,audio_path,label_path in tqdm(songs, desc="Loading..."):

        chroma = chromagram.getChroma(audio_path,chroma_type=chroma_type)
        df = pd.DataFrame(chroma,columns=chroma_labels)
        df['label'] = ''
        # parameters used by the deep Chroma Processors are 
        # NFFT=8192, HOPSIZE=4410, FS=44100 -> 10 frames/second
        df['time'] = np.linspace(0.1,len(chroma)*0.1,num=len(chroma))

        # label chroma features by iterating over all annotated chords
        labels_df = pd.read_csv(label_path,sep='\t',names=["tstart","tend","label"])
        for _,col in labels_df.iterrows():
            # access chroma features with time in chord annotation time interval 
            df.loc[(df['time'] >= float(col['tstart'])) & (df['time']< float(col['tend'])),
                        'label'] = col['label']
        chroma_df = pd.concat([chroma_df,df[labels]])
        chroma_df.reset_index(drop=True,inplace=True)
    return chroma_df

def getChordSequencesPaths(basepath):
    """ 
    This function returns a Pandas Dataframe with filepaths to all available audio,midi and annotation files 
    for the Chord_sequences Dataset (https://www.idmt.fraunhofer.de/en/publications/datasets/chord-sequences.html)  
    @basepath: path to the dataset
    """
    filepath_df = pd.DataFrame(columns=['ID','anchor.json','anchor.mid','anchor.wav',
                                    'negative.json','negative.mid','negative.wav',
                                    'positive.json','positive.mid','positive.wav'])
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
        col["ID"] = id
        col["anchor.json"] = os.path.join(basepath,id+"_anchor.json")
        col["anchor.mid"] = os.path.join(basepath,id+"_anchor.mid")
        col["anchor.wav"] = os.path.join(basepath,id+"_anchor.wav")
        col["negative.json"] = os.path.join(basepath,id+"_negative.json")
        col["negative.mid"] = os.path.join(basepath,id+"_negative.mid")
        col["negative.wav"] = os.path.join(basepath,id+"_negative.wav")
        col["positive.json"] = os.path.join(basepath,id+"_positive.json")
        col["positive.mid"] = os.path.join(basepath,id+"_positive.mid")
        col["positive.wav"] = os.path.join(basepath,id+"_positive.wav")
        filepath_df.loc[len(filepath_df),:] = col
    return filepath_df

def simplifyTriads(chordlabel):
    # remove remaining labels from chords
    try: 
        note,suffix = chordlabel.split(':')    
    except ValueError:
        note = chordlabel
        suffix = ""
    if note in enharmonic_notes:
        note = enharmonic_notes[note]
    if suffix.startswith('maj'):
        chordlabel = note
        return chordlabel
    elif suffix.startswith('min'):
        chordlabel = note+":min"
        return chordlabel
    elif suffix.startswith('aug'):
        chordlabel = note+":aug"
        return chordlabel
    elif suffix.startswith('dim'):
        chordlabel = note+":dim"
        return chordlabel
    return note

def simplifySevenths(chordlabel):
     # remove remaining labels from chords
    try: 
        note,suffix = chordlabel.split(':')
    except ValueError:
        note = chordlabel
        suffix = ""
    if note in enharmonic_notes:
        note = enharmonic_notes[note]

    if suffix.startswith('maj7'):
        # Major seventh chord
        chordlabel =  note+":maj7"
        return chordlabel
    elif suffix.startswith("maj"):
        # Major chord
        return note
    elif suffix.startswith('min7'):
        # Minor seventh chord
        chordlabel = note+":min7"       
        return chordlabel
    elif suffix.startswith("min"):
        chordlabel = note+":min"
        return chordlabel
    elif suffix.startswith('7'):
        # Dominant seventh chord
        chordlabel = note+':7'
        return chordlabel
    elif suffix.startswith('aug'):
        # Augmented chord
        chordlabel = note+":aug"        
        return chordlabel
    elif suffix.startswith('dim'):
        # Diminished Chord
        chordlabel = note+":dim"
        return chordlabel
    else:
        # Major chord
        return note

def simplifyAnnotations(df,chords="majmin"):
    df_copy = df.copy()
    if chords=="majmin":
        # remove all attached information from the label (slash chord etc.)
        df_copy['label'] = df['label'].apply(simplifyTriads)
    elif chords=="sevenths":
        df_copy['label'] = df['label'].apply(simplifySevenths)
    return df_copy


if __name__ == '__main__':
    basepath_labels="/home/max/ET-TI/Masterarbeit/prototyping/data/beatles/annotations/chords/"
    basepath_audio = "/home/max/ET-TI/Masterarbeit/prototyping/data/beatles/audio/"

    songs = getBeatlesPaths(basepath_labels,basepath_audio)
    chroma_df = getBeatlesChroma(songs,'madmom')
    chroma_df.to_pickle("/home/max/ET-TI/Masterarbeit/prototyping/data/beatles/beatles_chroma.pkl")  

    chroma_majmin_df = simplifyAnnotations(chroma_df,'majmin')
    chroma_majmin_df.to_pickle("/home/max/ET-TI/Masterarbeit/prototyping/data/beatles/beatles_chroma_majmin.pkl")  

