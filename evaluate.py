import ipywidgets
import IPython.display
import h5py
import os
import plots
import utils
import numpy as np

class hdf5GUI():
    """A simple ipython GUI to visualize data of a result file """
    def __init__(self,filepath=""):
        self.path = filepath
        self.current_figure = None
        self.result_file = None
        self.initializeGUI()

    def initializeGUI(self):
        self.output = ipywidgets.Output()
        self.dropdown_resultfile = ipywidgets.Dropdown(options=os.listdir(self.path),
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

    def load_result_file(self,*args):
        IPython.display.display(self.filepaths,self.selection,self.output,clear=True)
        try:
            self.result_file = h5py.File(os.path.join(self.path,self.dropdown_resultfile.value),"r")
        except Exception as e:
            print(f"error loading file:{os.path.join(self.path,self.dropdown_resultfile.value)}\n{e}") 
        with self.output:
            self.output.clear_output()
            self.preview_result_file()
            self.dropdown_dataset.options = [x for x in self.result_file] # datasets are groups of hdf5 files
            self.dropdown_dataset.value = self.dropdown_dataset.options[0]
            self.update_dataset() # update dropdown widgets

    def preview_result_file(self):
        """prints out an overview of the groups and subgroups of the loaded hdf5 file"""
        print("Metadata:")
        for m in self.result_file.attrs.keys():
            print(f"{m} -> {self.result_file.attrs[m]}")
        print("\nDatasets:\n    Track_ids:\n")
        for grp in self.result_file:
            print(f"- {grp}")
            n = 0
            for subgrp in self.result_file[grp]:
                print(f"    - {subgrp}")
                for subsubgrp in self.result_file[f"{grp}/{subgrp}"]:
                    print(f"      {subsubgrp}")
                n += 1
                if n > 2:
                    print(f"    :")
                    break

    def load_track(self,*args):
        if self.dropdown_resultfile.value is not None:
            IPython.display.display(self.filepaths,self.selection,self.output,clear=True)
            with self.output:
                self.output.clear_output()
                subgrp = self.result_file[f"{self.dropdown_dataset.value}/{self.dropdown_id.value}"]

                # print out metadata for track
                for k,v in subgrp.attrs.items():
                    print(k,v)
                
                # plot results
                chroma = subgrp.get("chroma")
                t_chroma = utils.timeVector(chroma.shape[1],hop_length=chroma.attrs.get("hop_length"))
                chroma_prefiltered = subgrp.get("chroma_prefiltered")
                ref_intervals = subgrp.get("ref_intervals")
                ref_labels = subgrp.get("ref_labels")
                # majmin_intervals = subgrp.get("majmin_intervals")
                # majmin_labels = subgrp.get("majmin_labels")
                # sevenths_intervals = subgrp.get("sevenths_intervals")
                # sevenths_labels = subgrp.get("sevenths_labels")
                # convert labels 
                ref_labels = [x.decode("utf-8") for x in ref_labels]
                # majmin_labels = [x.decode("utf-8") for x in majmin_labels]
                # sevenths_labels = [x.decode("utf-8") for x in sevenths_labels]
                plots.plotResults(t_chroma,chroma,chroma_prefiltered,ref_intervals,ref_labels,ref_intervals,ref_labels)

    def update_dataset(self,*args):
        """callback function for dropdown_dataset"""
        self.update_dropdown_id_options() # load new track_ids when dataset changed
        self.dropdown_id.value = self.dropdown_id.options[0] # set default track_id when changing dataset

    def update_dropdown_id_options(self,*args):
        self.dropdown_id.options = [x for x in self.result_file[self.dropdown_dataset.value]]
        self.update_selected_track_id()

    def update_selected_track_id(self,*args):
        if self.dropdown_id.value is not None:
            self.textbox_track_id.value = str(self.dropdown_id.value)
        else:
            self.textbox_track_id.value = "None"


if __name__ == "__main__":
    test_path = "/home/max/ET-TI/Masterarbeit/chord-recognition/results/test.hdf5"
    metadata= {"vocabulary":"majmin","algorithm":"template","prefilter":"median","postfilter":"hmm","datasets":["beatles","rw"]}
    track_ids = ["1","2","3","4","5","6"]
    chroma = np.zeros((12,100),dtype=float)
    with h5py.File(test_path,"w") as file:
        file.attrs.update(metadata)
        for dset in metadata["datasets"]:
            grp = file.create_group(dset)
            for track_id in track_ids:
                subgrp = grp.create_group(track_id)
                subsubgrp = subgrp.create_dataset("chroma",data=chroma)
                subsubgrp.attrs.create("nCRP",data=33)
                subsubgrp.attrs.create("sampleRate",data=22050)
                subsubgrp.attrs.create("hopSize",data=2048)
                subsubgrp = subgrp.create_dataset("chroma_prefiltered",data=chroma)
                subsubgrp.attrs.create("median_length",data=7)
