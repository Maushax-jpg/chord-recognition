import ipywidgets
import IPython.display as display
import h5py
import os
import plots
import utils
import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt

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
        chromadata = t_chroma,np.copy(chroma),np.copy(subgrp.get("chroma_prefiltered",chroma))
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
        plt.ioff() # interactive mode off

    def initializeGUI(self):
        # controls for selecting a file
        # TODO: check if valid resultfiles can be found in the given folder!
        # TODO: also check for subfolders!
        self.dropdown_resultfile = ipywidgets.Dropdown(options=os.listdir(self.path),
                                            value = None,description='File:',
                                            layout=ipywidgets.Layout(width='50%'),
                                            disabled=False)
        self.button_loadfile = ipywidgets.Button(description='Load File')

        # controls for selecting a track
        self.dropdown_dataset = ipywidgets.Dropdown(options=["beatles","rwc_pop","rw","queen"],
                                            value = "beatles",description='Dataset:',
                                            layout=ipywidgets.Layout(width='20%'),
                                            disabled=True)
        self.dropdown_id = ipywidgets.Dropdown(description='Track ID:',disabled=True,layout=ipywidgets.Layout(width='20%'))
        self.textbox_name = ipywidgets.Text(description='',disabled=True)
        
        # controls for displaying a plot
        self.t_start_slider = ipywidgets.IntSlider(min=0,step=5,readout_format='d',description="T0:")
        self.button_plot = ipywidgets.Button(description='create Plot')

        # create layout
        self.filepaths = ipywidgets.HBox([self.dropdown_resultfile,self.button_loadfile])
        self.selection = ipywidgets.HBox([self.dropdown_dataset, self.dropdown_id,self.textbox_name,self.t_start_slider, self.button_plot])
        # register callback functions
        self.button_loadfile.on_click(self.load_result_file)
        self.dropdown_dataset.observe(self.update_dataset,'value')
        self.dropdown_id.observe(self.update_selected_track_id, 'value')
        self.button_plot.on_click(self.plot_results)
        self.t_start_slider.observe(self.plot_results,"value")

        display.display(self.filepaths,self.selection)
        self.output_handle = display.display(ipywidgets.Output(),display_id=True)

    def plot_results(self,*args):
        plt.close("all")
        current_figure = plots.plotResults(self.chromadata,self.ground_truth,self.est_majmin,self.est_sevenths,
                        time_interval=(self.t_start_slider.value,self.t_start_slider.value+20)
                        ) 
        self.output_handle.update(current_figure)

    def load_result_file(self,*args):
        if self.dropdown_resultfile.value is None:
            self.output_handle.update(print("please select file!"))
            return
        try:
            self.dropdown_id.disabled = False
            self.dropdown_dataset.disabled = False
            self.trackdata_list,self.datasets = load_results(os.path.join(self.path,self.dropdown_resultfile.value))
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
        majmin_f = [x.majmin_f for x in self.trackdata_list]
        majmin_wscr =  [x.majmin_wscr for x in self.trackdata_list]
        majmin_seg = [x.majmin_seg for x in self.trackdata_list]
        sevenths_f = [x.sevenths_f for x in self.trackdata_list]
        sevenths_wscr =  [x.sevenths_wscr for x in self.trackdata_list]
        sevenths_seg = [x.sevenths_seg for x in self.trackdata_list]
        table_md = "### Evaluation Results: Combined dataset\n"
        table_md += "|eval-scheme| f-score| WSCR| Segmentation |\n| --- | --- |--- |--- |\n"
        table_md += f"|majmin|{100*np.mean(majmin_f):0.2f}+/-{100*np.std(majmin_f):0.2f}|{100*np.mean(majmin_wscr):0.2f}+/-{100*np.std(majmin_wscr):0.2f}"
        table_md += f"|{100*np.mean(majmin_seg):0.2f}+/-{100*np.std(majmin_seg):0.2f}|\n"
        table_md += f"|sevenths|{100*np.mean(sevenths_f):0.2f}+/-{100*np.std(sevenths_f):0.2f}|{100*np.mean(sevenths_wscr):0.2f}+/-{100*np.std(sevenths_wscr):0.2f}"
        table_md += f"|{100*np.mean(sevenths_seg):0.2f}+/-{100*np.std(sevenths_seg):0.2f}|\n"
        
        self.output_handle.update(display.Markdown(table_md))

    def display_trackdata(self):
        table_md = f"### {self.trackdata.name}\n"
        table_md += "|eval-scheme| f-score| WSCR| Segmentation |\n| --- | --- |--- |--- |\n"
        table_md += f"|majmin|{100*self.trackdata.majmin_f:0.2f}|{100*self.trackdata.majmin_wscr:0.2f}|{100*self.trackdata.majmin_seg:0.2f}|\n"
        table_md += f"|sevenths|{100*self.trackdata.sevenths_f:0.2f}|{100*self.trackdata.sevenths_wscr:0.2f}|{100*self.trackdata.sevenths_seg:0.2f}|\n"
        self.output_handle.update(display.Markdown(table_md))

    def load_track(self):
        self.trackdata,self.chromadata,self.ground_truth,self.est_majmin,self.est_sevenths = load_trackdata(os.path.join(self.path,self.dropdown_resultfile.value),
                                                                            self.dropdown_id.value,self.dropdown_dataset.value)
        # update t_max of slider 
        self.t_start_slider.max = self.chromadata[0][-1]-15 # access track length via t_chroma from chromadata
        self.display_trackdata()

    def update_dataset(self,*args):
        """callback function for dropdown_dataset"""
        self.update_dropdown_id_options() # load new track_ids when dataset changed
        self.dropdown_id.value = None # set default track_id when changing dataset

    def update_dropdown_id_options(self,*args):
        self.dropdown_id.options = [x.track_id for x in self.trackdata_list if x.dataset == self.dropdown_dataset.value]
        self.update_selected_track_id()

    def update_selected_track_id(self,*args):
        if self.dropdown_id.value is not None:
            self.load_track()
            self.textbox_name.value = self.trackdata.name
        else:
            self.textbox_name.value = "None"
