import ipywidgets
import IPython.display
import h5py
import os
import json

class hdf5GUI():
    """A simple ipython GUI to visualize data of a result file """
    def __init__(self,filepath=""):
        self.path = filepath
        self.current_figure = None
        self.result_file = None
        self.initializeGUI()

    def initializeGUI(self):
        self.output = ipywidgets.Output()
        self.dropdown_resultfile = ipywidgets.Dropdown(options=self.getResultFilepaths(),
                                            value = None,description='select resultfile:',
                                            layout=ipywidgets.Layout(width='50%'),
                                            disabled=False)
        self.button_loadfile = ipywidgets.Button(description='Load File')

        self.dropdown_dataset = ipywidgets.Dropdown(options=["beatles","rwc_pop","rw","queen"],
                                            value = "beatles",description='Dataset:',
                                            layout=ipywidgets.Layout(width='20%'),
                                            disabled=False)
        self.dropdown_id = ipywidgets.Dropdown(description='Track ID:',disabled=False,layout=ipywidgets.Layout(width='20%'))
        self.textbox_track_id = ipywidgets.Text(description='',disabled=True)
        self.button_load = ipywidgets.Button(description='Load Track')

        self.filepaths = ipywidgets.HBox([self.dropdown_resultfile,self.button_loadfile])
        self.selection = ipywidgets.HBox([self.dropdown_dataset, self.dropdown_id,self.textbox_track_id, self.button_load])

        # register callback functions
        self.button_loadfile.on_click(self.load_result_file)
        self.dropdown_dataset.observe(self.update_dataset,'value')
        self.dropdown_id.observe(self.update_selected_track_id, 'value')
        self.button_load.on_click(self.load_track)
        IPython.display.display(self.filepaths,self.output)
        
    def getResultFilepaths(self):
        return os.listdir(self.path)

    def load_result_file(self,*args):
        if self.dropdown_resultfile.value is not None:
            IPython.display.display(self.filepaths,self.selection,self.output,clear=True)
            with self.output:
                self.output.clear_output()
                try:
                    self.result_file = h5py.File(os.path.join(self.path,self.dropdown_resultfile.value),"r")
                    self.preview_result_file()
                except Exception as e:
                    print(f"error loading file:{os.path.join(self.path,self.dropdown_resultfile.value)}\n{e}")
        else:
            with self.output:
                self.output.clear_output()
                print("please select a result file")

    def preview_result_file(self):
        if "parameters" in self.result_file:
            params = json.loads(self.result_file['/parameters'][()])
            for key,value in params.items():
                print(key,value)

    
    def load_f_scores(self,filepath,dataset,return_params=False):
        """load f-scores for all tracks in a dataset from .hdf5 file specified by filepath """
        with h5py.File(filepath, 'r') as file:
            f_scores = []
            for track_id in file[f"/{dataset}"]:   
                track_data = file[f"/{dataset}/{track_id}"]
                majmin_score,seg_score = track_data.attrs.get("majmin"),track_data.attrs.get("seg")
                f_scores.append((2 * majmin_score * seg_score) / (majmin_score + seg_score))

            if return_params:
                params = json.loads(file['/parameters'][()])
                return f_scores, params
            else:
                return f_scores
    def load_track(self,*args):
        if self.dropdown_resultfile.value is not None:
            IPython.display.display(self.filepaths,self.selection,self.output,clear=True)
            with self.output:
                self.output.clear_output()
                print("load track")

    def update_dataset(self,*args):
        self.update_dropdown_id_options()        
        self.update_selected_track_id()

    def update_dropdown_id_options(self,*args):
        pass

    def update_selected_track_id(self,*args):
        pass

        
    def getSelectedTrack(self):
        return None