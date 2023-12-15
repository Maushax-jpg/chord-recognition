import ipywidgets
from IPython.display import display,Markdown
import h5py
import os
import plots
import utils
import numpy as np
from collections import namedtuple

trackdata = namedtuple('track','track_id name dataset majmin_wscr majmin_seg majmin_f sevenths_wscr sevenths_seg sevenths_f')
"""named tuple to store track specific metadata"""

def load_results(filepath):
    """reads all scores in a .hdf5 file and returns a list of trackdata and a list of the used datasets"""
    results = []
    with  h5py.File(filepath,"r") as file:
        datasets = file.attrs.get("dataset")

        if datasets is None:
            raise KeyError("Corrupt result file! no datasets are specified in the header!")
        
        if not isinstance(datasets,list): # convert to list if necessary
            datasets = list(datasets)

        for grp_name in datasets:
            for subgrp_name in file[f"{grp_name}/"]:
                subgrp = file[f"{grp_name}/{subgrp_name}"]
                results.append(trackdata( 
                        track_id=subgrp_name,
                        name=subgrp.attrs.get("name"),
                        dataset=grp_name,
                        majmin_wscr=subgrp.attrs.get("majmin_score"),
                        majmin_seg=subgrp.attrs.get("majmin_segmentation"),
                        majmin_f=subgrp.attrs.get("majmin_f"),
                        sevenths_wscr=subgrp.attrs.get("sevenths_score"),
                        sevenths_seg=subgrp.attrs.get("sevenths_segmentation"),
                        sevenths_f=subgrp.attrs.get("sevenths_f")
                    )
                )
    return results,datasets


def load_trackdata(filepath,track_id,dataset):
    """reads out trackdata, chromagram and annotations for a given track from an .hdf5 file"""
    with h5py.File(filepath,"r") as file:
        subgrp = file[f"{dataset}/{track_id}"]

        track_data = trackdata( 
                        track_id=track_id,
                        name=subgrp.attrs.get("name"),
                        dataset=dataset,
                        majmin_wscr=subgrp.attrs.get("majmin_score"),
                        majmin_seg=subgrp.attrs.get("majmin_segmentation"),
                        majmin_f=subgrp.attrs.get("majmin_f"),
                        sevenths_wscr=subgrp.attrs.get("sevenths_score"),
                        sevenths_seg=subgrp.attrs.get("sevenths_segmentation"),
                        sevenths_f=subgrp.attrs.get("sevenths_f")
                    )
        
        # load chroma to create time_vector from dataset
        chroma = subgrp.get("chroma")
        t_chroma = utils.timeVector(chroma.shape[1],hop_length=chroma.attrs.get("hop_length"))
        
        # strings have to be decode with utf-8 
    
        ref_labels = [x.decode("utf-8") for x in subgrp.get("ref_labels")]
        majmin_labels = [x.decode("utf-8") for x in subgrp.get("majmin_labels")]
        sevenths_labels = [x.decode("utf-8") for x in subgrp.get("sevenths_labels")]

        # convert to numpy arrays and pack to tuples
        chromadata = t_chroma,np.copy(chroma),np.copy(subgrp.get("chroma_prefiltered"))
        ground_truth = np.copy(subgrp.get("ref_intervals")), ref_labels
        est_majmin = np.copy(subgrp.get("majmin_intervals")),majmin_labels
        est_sevenths = np.copy(subgrp.get("sevenths_intervals")), sevenths_labels
    return track_data,chromadata,ground_truth,est_majmin,est_sevenths

class hdf5GUI():
    """A simple ipython GUI to visualize data of a result file """
    def __init__(self,filepath=""):
        self.path = filepath
        self.current_figure = None
        self.trackdata = []
        self.datasets = []
        self.initializeGUI()

    def initializeGUI(self):
        self.output = ipywidgets.Output()
        self.dropdown_resultfile = ipywidgets.Dropdown(options=os.listdir(self.path),
                                            value = None,description='File:',
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
        display(self.filepaths,self.output)

    def load_result_file(self,*args):
        display(self.filepaths,self.selection,self.output,clear=True)
        self.output.clear_output()
        try:
            self.trackdata,self.datasets = load_results(os.path.join(self.path,self.dropdown_resultfile.value))
            self.dropdown_dataset.options = self.datasets
            self.dropdown_dataset.value = self.dropdown_dataset.options[0]
            self.update_dataset() # updates all dropdown widgets
            self.preview_result_file()

        except KeyError:
            print("Corrupt File: No Metadata indicating used datasets available..")
        except FileNotFoundError:
            print("File not found.")
        except IOError as e:
            print(f"An error occurred while processing the file: {e}")


    def preview_result_file(self):
        """prints out an overview of the groups and subgroups of the loaded hdf5 file"""
        majmin_f = [x.majmin_f for x in self.trackdata]
        majmin_wscr =  [x.majmin_wscr for x in self.trackdata]
        majmin_seg = [x.majmin_seg for x in self.trackdata]

        sevenths_f = [x.sevenths_f for x in self.trackdata]
        sevenths_wscr =  [x.sevenths_wscr for x in self.trackdata]
        sevenths_seg = [x.sevenths_seg for x in self.trackdata]
        table_md = "# Evaluation Results on the combined dataset\n\n"
        table_md += "|eval-scheme| f-score| WSCR| Segmentation |\n| --- | --- |--- |--- |\n"
        table_md += f"|majmin|{np.mean(majmin_f):0.2f}+/-{np.std(majmin_f):0.2f}|{np.mean(majmin_wscr):0.2f}+/-{np.std(majmin_wscr):0.2f}"
        table_md += f"|{np.mean(majmin_seg):0.2f}+/-{np.std(majmin_seg):0.2f}|\n"
        table_md += f"|sevenths|{np.mean(sevenths_f):0.2f}+/-{np.std(sevenths_f):0.2f}|{np.mean(sevenths_wscr):0.2f}+/-{np.std(sevenths_wscr):0.2f}"
        table_md += f"|{np.mean(sevenths_seg):0.2f}+/-{np.std(sevenths_seg):0.2f}|\n"
        with self.output:
            display(Markdown(table_md))


    def load_track(self,*args):
        if self.dropdown_resultfile.value is not None:
            display(self.filepaths,self.selection,self.output,clear=True)
            trackdata,chromadata,ground_truth,est_majmin,est_sevenths = load_trackdata(os.path.join(self.path,self.dropdown_resultfile.value),
                                                                             self.dropdown_id.value,self.dropdown_dataset.value)
            with self.output:
                self.output.clear_output()
                table_md = f"## Evaluation results for {trackdata.name}\n\n"
                table_md += "|eval-scheme| f-score| WSCR| Segmentation |\n| --- | --- |--- |--- |\n"
                table_md += f"|majmin|{trackdata.majmin_f:0.2f}|{trackdata.majmin_wscr:0.2f}|{trackdata.majmin_seg:0.2f}|\n"
                table_md += f"|sevenths|{trackdata.sevenths_f:0.2f}|{trackdata.sevenths_wscr:0.2f}|{trackdata.sevenths_seg:0.2f}|\n"
                display(Markdown(table_md))
                plots.plotResults(chromadata[0],chromadata[1],chromadata[2],ground_truth[0],ground_truth[1],est_majmin[0],est_majmin[1])

    def update_dataset(self,*args):
        """callback function for dropdown_dataset"""
        self.update_dropdown_id_options() # load new track_ids when dataset changed
        self.dropdown_id.value = self.dropdown_id.options[0] # set default track_id when changing dataset

    def update_dropdown_id_options(self,*args):
        self.dropdown_id.options = [x.track_id for x in self.trackdata if x.dataset == self.dropdown_dataset.value]
        self.update_selected_track_id()

    def update_selected_track_id(self,*args):
        if self.dropdown_id.value is not None:
            self.textbox_track_id.value = str(self.dropdown_id.value)
            # print out name
        else:
            self.textbox_track_id.value = "None"
