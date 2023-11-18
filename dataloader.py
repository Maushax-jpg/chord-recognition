import os
from torch.utils.data import Dataset
import mir_eval
import mirdata
import json
import ipywidgets
import IPython.display

class MIRDataset(Dataset):
    """
    Dataloader class for MIR Datasets. Available Datasets are "rwc_popular" and "beatles"
    """
    def __init__(self,name,basepath="/home/max/ET-TI/Masterarbeit/mirdata/",split_nr=1):
        super().__init__()
        self._path = basepath
        self._split_nr = split_nr
        self._name = name
        self._dataset = self.loadDataset(name)
        self._tracks = self._dataset.load_tracks()

    def __len__(self):
        """returns rows of dataframe"""
        return len(self._dataset.track_ids)
    
    def __getitem__(self, track_id):
        target = self.getAnnotations(track_id)
        audio_path = self._tracks[track_id].audio_path
        # replace "' with _"
        audio_path = audio_path.replace("'","_")
        return audio_path, target
    
    def getTrackIDs(self):
        """returns list of available track IDs that can be used to access the dataset tracks"""
        return self._dataset.track_ids
    
    def getTitle(self,track_id):
        return self._tracks[track_id].title
    
    def getTrackList(self):
        """returns list of available track names"""
        with open(f"{self._path}{self._name}/splits/split_{self._split_nr}.json", 'r', encoding='UTF-8') as file:
            track_list = json.load(file)
        return track_list

    def loadDataset(self,name):
        return mirdata.initialize(name, data_home=os.path.join(self._path,name))
    
    def getAnnotations(self,track_id):
        if self._name == 'billboard':
            return None
        else:
            path = self._tracks[track_id].chords_path
            path = path.replace("'","_")  # replace if necessary!
            return  mir_eval.io.load_labeled_intervals(path)

class MIRDatasetGUI():
    """A simple GUI to select songs from a dataset """
    def __init__(self,path=""):
        self.path = path
        self.dataset = None
        self.initializeGUI()
        self.displayGUI()

    def initializeGUI(self):
        self.output = ipywidgets.Output()
        self.dropdown_dataset = ipywidgets.Dropdown(options=["beatles","rwc_popular"],value = "beatles",description='Dataset:',
                                  layout=ipywidgets.Layout(width='20%'),disabled=False)
        self.dropdown_split = ipywidgets.Dropdown(options=[1, 2, 3, 4, 5, 6, 7],value = 3,description='Split:',
                                        layout=ipywidgets.Layout(width='15%'),disabled=False)
        self.dropdown_id = ipywidgets.Dropdown(description='Track ID:',disabled=False,layout=ipywidgets.Layout(width='20%'))
        self.textbox_track_id = ipywidgets.Text(description='',disabled=True)
        self.button_load = ipywidgets.Button(description='Load Track')
        self.selection = ipywidgets.HBox([self.dropdown_dataset,self.dropdown_split, self.dropdown_id,self.textbox_track_id, self.button_load])

        # register callback functions
        self.dropdown_split.observe(self.update_dropdown_id_options, 'value')
        self.dropdown_id.observe(self.update_selected_track_id, 'value')
        self.update_dropdown_id_options()
        self.update_selected_track_id()
        
    def update_dropdown_id_options(self,*args):
        selected_split = self.dropdown_split.value
        self.dataset = MIRDataset(self.dropdown_dataset.value,basepath=self.path, split_nr=selected_split)
        self.dropdown_id.options = list(self.dataset.getTrackList().keys())
        self.dropdown_id.value = list(self.dataset.getTrackList().keys())[0]

    def update_selected_track_id(self,*args):
        self.textbox_track_id.value = self.dataset.getTrackList()[self.dropdown_id.value]

    def on_control_change(self,change):  
        # change some controls according to updates
        pass
        # plotData(range_slider.value,beat_alignment.value,chroma_type.value)

    def displayGUI(self):
        IPython.display.display(self.selection,self.output)

    def getSelectedTrack(self):
        target = self.dataset.getAnnotations(self.dropdown_id.value)
        audio_path = self.dataset._tracks[self.dropdown_id.value].audio_path
        return audio_path,target