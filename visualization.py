import ipywidgets
import IPython.display as display
import h5py
import os
import utils
import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt
import mir_eval
import librosa.display

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
        
        # strings have to be decoded with utf-8 
        ref_labels =      [x.decode("utf-8") for x in subgrp.get("ref_labels")]
        majmin_labels =   [x.decode("utf-8") for x in subgrp.get("majmin_labels")]
        sevenths_labels = [x.decode("utf-8") for x in subgrp.get("sevenths_labels")]

        # convert to numpy arrays and pack to tuples
        chromadata = t_chroma,np.copy(chroma),np.copy(subgrp.get("chroma_prefiltered",chroma))
        ground_truth = np.copy(subgrp.get("ref_intervals")), ref_labels
        est_majmin = np.copy(subgrp.get("majmin_intervals")),majmin_labels
        est_sevenths = np.copy(subgrp.get("sevenths_intervals")), sevenths_labels
    return track_data,chromadata,ground_truth,est_majmin,est_sevenths

class chromaApp():
    """an ipython app to visualize chromagram data"""
    def __init__(self,filepath=""):
        
        plt.ioff()
        self.loadChromadata(filepath)
        self.initializeGUI()

    def loadChromadata(self,filepath):
        try:
            file = h5py.File(filepath,"r")
        except FileNotFoundError as e:
            print(f"{e}\n\nPlease specify correct filepath!")
        self.chromadata = {}
        for group in file:
            try:
                root,qual,scale_deg,bass = group.split("-")
                label = mir_eval.chord.join(root,qual,eval(scale_deg),bass)
            except ValueError:
                label = "N"
            self.chromadata[label] = file[group].get(group)

    def initializeGUI(self):
        # sort dictionary depending on the size of the numpy array and store the labels as options
        labels = list(dict(sorted(self.chromadata.items(),key=lambda x:x[1].shape[1],reverse=True)).keys())
        self.dd_chords = ipywidgets.Dropdown(
            options=labels,
            value = labels[0],
            description='Chord:',
        )
        self.dd_template = ipywidgets.Dropdown(
            options=list(mir_eval.chord.QUALITIES.keys()),
            value="maj",
            description='Template:',
        )
        self.output = ipywidgets.Output()
        self.slider_time = ipywidgets.IntRangeSlider(
            min=0,
            description="select column Index:",
            layout=ipywidgets.Layout(width='30%')
        )
        self.dd_chords.observe(self.update_sliders,'value')
        self.dd_template.observe(self.plotChroma,'value')
        self.slider_time.observe(self.plotChroma,'value')
        
        display.display(ipywidgets.HBox((self.dd_chords,self.dd_template)),self.slider_time)
        self.display_handle = display.display(self.output,display_id=True)


    def update_sliders(self,*args):
        n_cols = self.chromadata[self.dd_chords.value].shape[1]
        self.slider_time.max = n_cols
        self.slider_time.value=(0,self.slider_time.max)

    def plotChroma(self,*args):
        plt.close("all")
        temp = self.chromadata[self.dd_chords.value][:,self.slider_time.value[0]:self.slider_time.value[1]]
        # check for No chord label -> there is no valid template
        if self.dd_chords.value == "N":
            fig,(ax2,ax0,ax1) = plt.subplots(1,3,width_ratios=(0.3,10,.3),figsize=(9,5))
            ax2.axis("off")
            img = librosa.display.specshow(temp,ax=ax0,hop_length=2048,y_axis='chroma', cmap="Reds",vmin=0, vmax=np.max(temp))
            plt.colorbar(img,ax1)
            ax1.set_ylabel("heatmap values")
            self.display_handle.update(fig)
            return

        # create a template chromavector as comparison measure to sort the array
        template_vector = np.array(mir_eval.chord.QUALITIES[self.dd_template.value])
        template_vector = template_vector / np.linalg.norm(template_vector)
        # Compute inner products between each vector and the template vector
        inner_products = np.dot(temp.T, template_vector)
        sorted_indices = np.argsort(inner_products)[::-1]
        sorted_chromadata = temp[:, sorted_indices]
        repeated_template = np.tile(template_vector, (12, 1)).T

        fig,(ax2,ax0,ax1) = plt.subplots(1,3,width_ratios=(0.3,10,.3),figsize=(9,5))
        img = librosa.display.specshow(repeated_template,ax=ax2, cmap="Reds",vmin=0, vmax=np.max(repeated_template))
        ax2.set_ylabel("Chord Template")
        img = librosa.display.specshow(sorted_chromadata,ax=ax0,hop_length=2048,y_axis='chroma', cmap="Reds",vmin=0, vmax=np.max(repeated_template))
        plt.colorbar(img,ax1)
        ax1.set_ylabel("heatmap values")
        ax = ax0.twinx()
        ax.plot(inner_products[sorted_indices], 'b', label='Inner Product Values')
        ax.set_ylabel('Inner Product Values', color='blue')
        ax.set_ylim(0,0.6)
        ax.tick_params('y', colors='blue')

        fig.tight_layout(pad=1)
        ax0.set_xlabel("Sorted Chromagram")
        self.display_handle.update(fig)    


class visualizationApp():
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
        self.dropdown_resultfile = ipywidgets.Dropdown(
            options=os.listdir(self.path),
            value = None,
            description='File:',
            layout=ipywidgets.Layout(width='40%'),
            disabled=False
        )

        self.button_loadfile = ipywidgets.Button(
            description='Load File'
        )

        # controls for selecting a track
        self.dropdown_dataset = ipywidgets.Dropdown(
            options=["beatles","rwc_pop","rw","queen"],
            value = "beatles",#
            description='Dataset:',
            layout=ipywidgets.Layout(width='10%'),
            disabled=True
        )

        self.dropdown_id = ipywidgets.Dropdown(
            description='Track ID:',
            disabled=True,
            layout=ipywidgets.Layout(width='30%')
        )

        self.button_plot = ipywidgets.Button(
            description='create Plot'
        )
        
        # controls for displaying a plot
        self.t_start_slider = ipywidgets.IntSlider(
            min=0,
            step=5,
            readout_format='d',
            description="T0:",
            disabled=True,
            layout=ipywidgets.Layout(width='30%')
        )
        self.delta_t = ipywidgets.IntText(
            value=20,
            description='dT:',
            disabled=True,
            layout=ipywidgets.Layout(width='10%')
        )
        # create layout
        self.filepaths = ipywidgets.HBox(
            [
                self.dropdown_resultfile,
                self.button_loadfile
            ]
        )
        self.selection = ipywidgets.HBox(
            [
                self.dropdown_dataset,
                self.dropdown_id, 
                self.button_plot
            ]
        )
        self.controls = ipywidgets.HBox(
            [
                self.t_start_slider,
                self.delta_t,
            ]   
        )

        # register callback functions
        self.button_loadfile.on_click(self.load_result_file)
        self.dropdown_dataset.observe(self.update_dataset,'value')
        self.dropdown_id.observe(self.update_selected_track_id, 'value')
        self.button_plot.on_click(self.plot_results)
        self.t_start_slider.observe(self.plot_results,"value")
        self.delta_t.observe(self.plot_results,"value")
        
        display.display(self.filepaths,self.selection,self.controls)
        self.output_handle = display.display(ipywidgets.Output(),display_id=True)


    def plot_results(self,*args):
        plt.close("all")
        t_chroma,chroma,chroma_prefiltered = self.chromadata
        ref_intervals,ref_labels = self.ground_truth
        est_majmin_intervals,est_majmin_labels = self.est_majmin
        est_sevenths_estimation_intervals,est_sevenths_estimation_labels = self.est_sevenths  

        # select chromadata / correlation or whatever
        time_interval = (self.t_start_slider.value, self.t_start_slider.value + self.delta_t.value)

        fig,((ax_1,ax_11),(ax_2,ax_21)) = plt.subplots(2,2,height_ratios=(5,5),width_ratios=(20,.3),figsize=(9,5))

        # create chord annotation plot
        utils.plotChordAnnotations(ax_1,ref_intervals,ref_labels,time_interval=time_interval,y_0=6)
        ax_1.text(time_interval[0],7.7,"Ground truth annotations")
        utils.plotChordAnnotations(ax_1,est_majmin_intervals,est_majmin_labels,time_interval=time_interval,y_0=3)
        ax_1.text(time_interval[0],4.7,"Estimated with majmin alphabet")
        utils.plotChordAnnotations(ax_1,est_sevenths_estimation_intervals,est_sevenths_estimation_labels,time_interval=time_interval,y_0=0)
        ax_1.text(time_interval[0],1.7,"Estimated with sevenths alphabet")
        ax_1.set_ylim(0,8)
        ax_1.set_yticks([])
        # Hide the y-axis line
        ax_1.spines['left'].set_visible(False)
        ax_1.spines['right'].set_visible(False)
        ax_1.spines['top'].set_visible(False)
        ax_1.axis("on")
        ax_11.set_axis_off()

        # plot prefiltered chromagram
        img = utils.plotChromagram(ax_2,t_chroma,chroma_prefiltered,time_interval=time_interval)
        fig.colorbar(img,cax=ax_21)

        # create a label for the estimates
        xticks = np.linspace(time_interval[0],time_interval[1],21)
        xticklabels = [xticks[i] if i % 5 == 0 else "" for i in range(21)]
        for ax in [ax_1,ax_2]:
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels)
            ax.set_xlabel("")

        ax_2.set_xlabel("Time in s")
        fig.tight_layout(h_pad=0.1,w_pad=0.1,pad=0.3)
        self.output_handle.update(fig)

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
        self.t_start_slider.max = self.chromadata[0][-1]-self.delta_t.value # access track length via t_chroma from chromadata
        self.t_start_slider.disabled = False
        self.delta_t.disabled = False
        self.display_trackdata()

    def update_dataset(self,*args):
        """callback function for dropdown_dataset"""
        self.update_dropdown_id_options() # load new track_ids when dataset changed

    def update_dropdown_id_options(self,*args):
        self.dropdown_id.value = None
        self.dropdown_id.options = [(x.name,x.track_id) for x in self.trackdata_list if x.dataset == self.dropdown_dataset.value]
        self.update_selected_track_id()

    def update_selected_track_id(self,*args):
        if self.dropdown_id.value is not None:
            self.load_track()
