import os
import re
import pandas as pd
import tqdm

def getBeatlesPaths(basepath_labels,basepath_audio):
    '''This function returns a list of tuples containing songname and paths to .mp3 and .chords files
        basepath_labels: path to chord annotations
        basepath_audio: path to beatles albums 
    '''
    # this code snippet searches for a matching annotation file in basepath_labels for every song found in
    # basepath_audio. The filenames are not named with the same syntax so regular expressions are used to 
    # find the right filenames   
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
            songname = re.sub(r"[\'\.\!\,]", "", temp)
            # search correct file with labels for the song
            found= False
            for labelfile in labelfiles:
                if labelfile.lower().startswith(songname.lower()):
                    audiopath = os.path.join(folderpath,file)
                    labelpath = os.path.join(basepath_labels,labelfile)
                    songs.append((songname,audiopath,labelpath)) 
                    found = True
                    break
            if not found:
                print(f"songname not found: {songname}")
    return songs

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
    data = {}
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

def simplyfyAnnotations(df,type="majmin"):
    df_copy = df.copy()
    if type=="majmin":
        # remove all attached information from the label (slash chord etc.)
        for row,col in tqdm(df.iterrows(),desc="Processing.."):
            pattern = re.compile(r"([A-Za-z]#?:min)|([A-Za-z]#?:maj)")
            match = pattern.search(col["label"])
            if match:
                df_copy.at[row,"label"]= match.group()
    elif type=="sevenths":
        # remove all attached information from the label (slash chord etc.)
        for row,col in tqdm(df.iterrows(),desc="Processing.."):
            pattern = re.compile(r"([A-Za-z]#?(:min)?7?)|([A-Za-z]#?(:maj)?7?)|([A-Za-z]#?(:7)?)")
            match = pattern.search(col["label"])
            if match:
                df_copy.at[row,"label"]= match.group()
    return df_copy