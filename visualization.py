import ipywidgets
import IPython.display as display
import h5py
import os
import utils
import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt
import librosa.display


trackdata = namedtuple('track','track_id dset name')
"""named tuple to store track specific metadata"""

experiments =  [("source separation vocals",["source_separation_vocals.hdf5", "source_separation_none.hdf5"]),
                ("source separation drums",["source_separation_drums.hdf5", "source_separation_none.hdf5"]),
                ("source separation both",["source_separation_both.hdf5", "source_separation_none.hdf5"]),
                ("prefilter",["prefilter_rp.hdf5","prefilter_median.hdf5"]),
                ("distance metrics",["distance_measure.hdf5"]),
                ("pitchspace",["crp_pitchspace.hdf5"]),
                ("deepchroma",["crp_pitchspace.hdf5", "dcp_pitchspace.hdf5","dcp_madmom.hdf5"])]


def load_tracklist(filepath):
    """reads all scores in a .hdf5 file and returns a list of trackdata and a list of the used datasets"""
    results = []
    with  h5py.File(filepath,"r") as file:
        datasets = file.attrs.get("dataset",default=["beatles","rwc_pop","rw","queen"])
        for grp_name in datasets:
            for subgrp_name in file[f"{grp_name}/"]:
                subgrp = file[f"{grp_name}/{subgrp_name}"]
                results.append(trackdata( 
                        track_id=subgrp_name,
                        dset=grp_name,
                        name=subgrp.attrs.get("name")
                    )
                )
    return results,datasets

def load_trackdata(filepath,track_id,dataset,key_intervals,key_labels):
    """reads out trackdata, chromagram and annotations for a given track from an .hdf5 file"""
    with h5py.File(filepath,"r") as file:
        subgrp = file[f"{dataset}/{track_id}"]
     
        # load chroma to create time_vector from dataset
        chroma = subgrp.get("chroma")
        t_chroma = utils.timeVector(chroma.shape[1],hop_length=chroma.attrs.get("hop_length"))
        # strings have to be decoded with utf-8 
        ref_labels =      [x.decode("utf-8") for x in subgrp.get("ref_labels")]
        try:
            labels =   [x.decode("utf-8") for x in subgrp[key_labels]]
            intervals = np.copy(subgrp[key_intervals])
        except KeyError:
            print([x for x in subgrp])
            raise ValueError
        # convert to numpy arrays and pack to tuples
        cqt =  np.copy(subgrp.get("pitchgram_cqt",np.array([])))
        data = t_chroma,cqt, np.copy(subgrp.get("chroma_prefiltered",chroma)),np.copy(subgrp.get("hcdf"))
        ground_truth = np.copy(subgrp.get("ref_intervals")), ref_labels
    return data, ground_truth, intervals, labels


class visualizationApp():
    """A simple ipython GUI to visualize data of a result file """
    def __init__(self,filepath):
        self.path = filepath # path to result files
        self.current_figure = None
        self.chromadata = None
        self.trackdata = []
        self.datasets = []
        self.initializeGUI()

    def initializeGUI(self):
        # controls for selecting a file
        self.dropdown_resultfile = ipywidgets.Dropdown(
            options=experiments,
            value = None,
            description='Experiment:',
            layout=ipywidgets.Layout(width='80%'),
            disabled=False)

        # controls for selecting a track
        self.dropdown_dataset = ipywidgets.Dropdown(
            options=["beatles","rwc_pop","rw","queen"],
            value = "beatles",#
            description='Dataset:',
            layout=ipywidgets.Layout(width='30%'),
            disabled=True)

        self.dropdown_id = ipywidgets.Dropdown(
            description='Track ID:',
            disabled=True,
            layout=ipywidgets.Layout(width='40%'))
        
        self.text = ipywidgets.Text(
            value='-',
            placeholder='',
            description='',
            disabled=True,
            layout=ipywidgets.Layout(width='10%'))
        
        # controls for displaying a plot
        self.t_start_slider = ipywidgets.IntSlider(
            min=0,
            step=5,
            readout_format='d',
            description="T0:",
            disabled=True,
            layout=ipywidgets.Layout(width='50%'))
        
        self.delta_t = ipywidgets.IntText(
            value=20,
            description='dT:',
            disabled=True,

            layout=ipywidgets.Layout(width='20%'))
        
        # create layout
        self.filepaths = ipywidgets.HBox(
            [
                self.dropdown_resultfile
            ]
        )
        self.selection = ipywidgets.HBox(
            [
                self.dropdown_dataset,
                self.dropdown_id, 
                self.text
            ]
        )
        self.controls = ipywidgets.HBox(
            [
                self.t_start_slider,
                self.delta_t
            ]   
        )

        # register callback functions
        self.dropdown_resultfile.observe(self.load_result_file,'value')
        self.dropdown_dataset.observe(self.update_dataset,'value')
        self.dropdown_id.observe(self.update_selected_track_id, 'value')
        self.dropdown_id.observe(self.plot_results,'value')
        self.t_start_slider.observe(self.plot_results,"value")
        self.delta_t.observe(self.plot_results,"value")
        display.display(self.filepaths,self.selection,self.controls)
        self.output_handle = display.display(ipywidgets.Output(),display_id=True)

    def plot_pitchspace_results(self,*args):
        plt.close("all")
        t_chroma,cqt,chroma,hcdf = self.chromadata
        # select chromadata / correlation or whatever
        time_interval = (self.t_start_slider.value, self.t_start_slider.value + self.delta_t.value)

        fig,((ax_1,ax_11),(ax_2,ax_21),(ax_3,ax_31)) = plt.subplots(3,2,height_ratios=(3,2,9),width_ratios=(20,.3),figsize=(7,9))
        # create chord annotation plot
        self.plot_annotations(ax_1, ax_11, time_interval)

         # plot prefiltered chromagram
        img = utils.plotChromagram(ax_2,t_chroma,chroma,time_interval=time_interval)
        fig.colorbar(img,cax=ax_21)
        i0,i1 = utils.getTimeIndices(t_chroma,time_interval)
        # ax_3.plot(t_chroma[i0:i1],hcdf[i0:i1])
        self.align_time_axes([ax_1,ax_2,ax_3],time_interval)
        # plot cqt
        i0,i1 = utils.getTimeIndices(t_chroma,time_interval)
        img = librosa.display.specshow(librosa.amplitude_to_db(cqt[:,i0:i1],ref=np.max(cqt[:,i0:i1])),
                                    x_coords=t_chroma[i0:i1],
                                    x_axis="time",
                                    y_axis='cqt_note',
                                    cmap="viridis",
                                    fmin=librosa.midi_to_hz(36),
                                    bins_per_octave=12,
                                    ax=ax_3,
                                    vmin=-70,
                                    vmax=0)
        cbar = fig.colorbar(img,cax=ax_31)
        ax_3.set_ylim(librosa.midi_to_hz(36),librosa.midi_to_hz(102))
        cbar.ax.set_ylabel("dB", rotation=-90, va="bottom")
        self.align_time_axes([ax_1,ax_2,ax_3],time_interval)
    
        fig.tight_layout(h_pad=0.1,w_pad=0.1,pad=0.3)
        self.output_handle.update(fig)

    def plot_source_separation_results(self,*args):
        plt.close("all")
        time_interval = (self.t_start_slider.value, self.t_start_slider.value + self.delta_t.value)
        fig,((ax_1,ax_11),(ax_2,ax_21),(ax_3,ax_31)) = plt.subplots(3,2,height_ratios=(3,3,9),width_ratios=(20,.3),figsize=(7,9))
        
        # create chord annotation plot
        self.plot_annotations(ax_1, ax_11, time_interval)
        
        # plot prefiltered chromagram
        img = utils.plotChromagram(ax_2,self.chromadata[0],self.chromadata[2],time_interval=time_interval)
        fig.colorbar(img,cax=ax_21)

        # plot cqt
        i0,i1 = utils.getTimeIndices(self.chromadata[0],time_interval)
        img = librosa.display.specshow(librosa.amplitude_to_db(self.chromadata[1][:,i0:i1],
            ref=np.max(self.chromadata[1][:,i0:i1])),ax=ax_3,
            x_coords=self.chromadata[0][i0:i1],x_axis="time",y_axis='cqt_note',cmap="viridis",
            fmin=librosa.midi_to_hz(36),bins_per_octave=12, vmin=-70,vmax=0)
        cbar = fig.colorbar(img,cax=ax_31)
        cbar.ax.set_ylabel("dB", rotation=-90, va="bottom")
        self.align_time_axes([ax_1,ax_2,ax_3],time_interval)
        fig.tight_layout(h_pad=0.1,w_pad=0.1,pad=0.3)
        self.output_handle.update(fig)

    def plot_prefilter_results(self,*args):
        plt.close("all")
        t_chroma,_,chroma,_ = self.chromadata[0]
        _,_,chroma_median,_ = self.chromadata[1]

        time_interval = (self.t_start_slider.value, self.t_start_slider.value + self.delta_t.value)

        fig,((ax_1,ax_11),(ax_2,ax_21),(ax_3,ax_31)) = plt.subplots(3,2,height_ratios=(1.5,1,1),
                                                                    width_ratios=(20,.3),
                                                                    figsize=(7,5))
        self.plot_annotations(ax_1,ax_11,time_interval)

        # plot prefiltered chromagrams
        img = utils.plotChromagram(ax_2,t_chroma,chroma_median,time_interval=time_interval)
        fig.colorbar(img,cax=ax_21)
        ax_2.set_ylabel("RP prefilter")
        img = utils.plotChromagram(ax_3,t_chroma,chroma,time_interval=time_interval)
        fig.colorbar(img,cax=ax_31)
        ax_3.set_ylabel("Median prefilter")
        self.align_time_axes([ax_1,ax_2,ax_3],time_interval)
        fig.tight_layout(w_pad=0.1,pad=0.3)
        self.output_handle.update(fig)

    def plot_distance_metric_results(self,*args):
        plt.close("all")
        time_interval = (self.t_start_slider.value, self.t_start_slider.value + self.delta_t.value)
    
        fig,((ax_1,ax_11),(ax_2,ax_21),(ax_3,ax_31)) = plt.subplots(3,2,height_ratios=(3,9,3),width_ratios=(20,.3),figsize=(7,9))
        self.plot_annotations(ax_1,ax_11,time_interval)

        # plot prefiltered chromagram
        img = utils.plotChromagram(ax_3,self.chromadata[0],self.chromadata[2],time_interval=time_interval)
        fig.colorbar(img,cax=ax_31)
        # plot cqt
        i0,i1 = utils.getTimeIndices(self.chromadata[0],time_interval)
        img = librosa.display.specshow(librosa.amplitude_to_db(self.chromadata[1][:,i0:i1],ref=np.max(self.chromadata[1][:,i0:i1])),
            x_coords=self.chromadata[0][i0:i1],x_axis="time",y_axis='cqt_note',cmap="viridis",
            fmin=librosa.midi_to_hz(36),bins_per_octave=12,ax=ax_2,vmin=-70,vmax=0)
        ax_2.set_ylim(librosa.midi_to_hz(24),librosa.midi_to_hz(104))
        cbar = fig.colorbar(img,cax=ax_21)
        cbar.ax.set_ylabel("dB", rotation=-90, va="bottom")
        ax_2.set_ylabel("tuned Pitchgram - CQT")
        self.align_time_axes([ax_1, ax_2, ax_3],time_interval)

        fig.tight_layout(w_pad=0.1,pad=0.3)
        self.output_handle.update(fig)

    def plot_deepchroma_results(self,*args):
        plt.close("all")
        t_chroma_crp,cqt,crp,_ = self.chromadata[0]
        t_chroma_deepchroma,_,deepchroma,_ = self.chromadata[1]
        time_interval = (self.t_start_slider.value, self.t_start_slider.value + self.delta_t.value)

        fig,((ax_1,ax_11),(ax_2,ax_21),(ax_3,ax_31),(ax_4,ax_41)) = plt.subplots(4,2,height_ratios=(1,1,1,3),
                                                                    width_ratios=(20,.3),
                                                                    figsize=(7,9))
        self.plot_annotations(ax_1,ax_11,time_interval)

        # plot chromagrams
        img = utils.plotChromagram(ax_2,t_chroma_deepchroma,deepchroma,time_interval=time_interval)
        fig.colorbar(img,cax=ax_21)
        ax_2.set_ylabel("deep chroma")

        img = utils.plotChromagram(ax_3,t_chroma_crp,crp,time_interval=time_interval)
        fig.colorbar(img,cax=ax_31)
        ax_3.set_ylabel("CRP")

        # plot cqt
        i0,i1 = utils.getTimeIndices(t_chroma_crp,time_interval)
        img = librosa.display.specshow(librosa.amplitude_to_db(cqt[:,i0:i1],
            ref=np.max(cqt[:,i0:i1])),ax=ax_4,
            x_coords=t_chroma_crp[i0:i1],x_axis="time",y_axis='cqt_note',cmap="viridis",
            fmin=librosa.midi_to_hz(36),bins_per_octave=12, vmin=-70,vmax=0)
        cbar = fig.colorbar(img,cax=ax_41)
        ax_4.set_ylim(librosa.midi_to_hz(36),librosa.midi_to_hz(102))
        cbar.ax.set_ylabel("dB", rotation=-90, va="bottom")

        # align time axes and display plot
        self.align_time_axes([ax_1,ax_2,ax_3,ax_4],time_interval)
        fig.tight_layout(w_pad=0.1,pad=0.3)

        self.output_handle.update(fig)


    def plot_annotations(self, ax1, ax2, time_interval):
        # create chord annotation plot
        n = len(self.transcriptions)
        utils.plotChordAnnotations(ax1,self.ground_truth[0],self.ground_truth[1],time_interval=time_interval,y_0=2.5*n)
        ax2.text(0,2.5*n+0.5,"GT")
        for i,(description,intervals,labels) in enumerate(self.transcriptions):
            utils.plotChordAnnotations(ax1, intervals,labels, time_interval=time_interval, y_0=2.5*i)
            ax2.text(0,2.5*i+0.5,description)
        ax1.set_ylim(0,2.5*n+3)
        ax2.set_ylim(0,2.5*n+3)
        ax1.set_yticks([])
        # Hide the y-axis line
        ax1.spines['left'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['top'].set_visible(False)
        ax1.axis("on")
        ax2.set_axis_off()

    def align_time_axes(self, axes, time_interval):
        # create a label for the estimates
        xticks = np.linspace(time_interval[0],time_interval[1],21)
        xticklabels = [xticks[i] if i % 5 == 0 else "" for i in range(21)]
        for ax in axes:
            ax.set_xticks(xticks)
            ax.set_xlabel("")
            ax.set_xticklabels([])
        axes[-1].set_xticklabels(xticklabels)
        axes[-1].set_xlabel("Time in s")

    def load_result_file(self,*args):
        try:
            self.output_handle.update("Please select Track to create plots!")
            self.filepaths = [os.path.join(self.path,x) for x in self.dropdown_resultfile.value]
            self.dropdown_id.disabled = False
            self.dropdown_dataset.disabled = False
            self.tracklist,self.datasets = load_tracklist(self.filepaths[0])
            self.dropdown_dataset.options = self.datasets
            self.dropdown_dataset.value = self.dropdown_dataset.options[0]
            self.update_dataset() # updates all dropdown widgets
        except KeyError:
            print("Corrupt File: No Metadata indicating used datasets available..")
        except FileNotFoundError:
            print(f"File {self.filepaths[0]} not found.")

    def plot_results(self,*args):
        """wrapper function for experiment specific plots"""
        if self.dropdown_resultfile.label.startswith("source separation"):
            self.plot_source_separation_results()
        elif self.dropdown_resultfile.label == "prefilter":
            self.plot_prefilter_results()
        elif self.dropdown_resultfile.label == "distance metrics":
            self.plot_distance_metric_results()
        elif self.dropdown_resultfile.label == "pitchspace":
            self.plot_pitchspace_results()
        elif self.dropdown_resultfile.label == "deepchroma":
            self.plot_deepchroma_results()

    def load_track(self):
        """load transcriptions and chromadata depending on selected experiment"""
        if self.dropdown_resultfile.label.startswith("source separation"):
            self.transcriptions = []
            self.chromadata,self.ground_truth,intervals,labels = load_trackdata(self.filepaths[0],
                                self.dropdown_id.value,self.dropdown_dataset.value,
                                "majmin_intervals","majmin_labels")
            self.transcriptions.append((f'-{self.dropdown_resultfile.label.split(" ")[-1]}', intervals, labels))

            _,_,intervals,labels = load_trackdata(self.filepaths[1],
                                self.dropdown_id.value,self.dropdown_dataset.value,
                                "majmin_intervals","majmin_labels")
            self.transcriptions.append(("mix",intervals,labels))
            self.updateControls(self.chromadata[0][-1], self.chromadata[1])
            return
        elif self.dropdown_resultfile.label == "prefilter":
            self.transcriptions = []
            self.chromadata = []
            median_chromadata,self.ground_truth,intervals,labels = load_trackdata(self.filepaths[1],
                    self.dropdown_id.value,self.dropdown_dataset.value,
                    "majmin_intervals","majmin_labels")
            self.chromadata.append(median_chromadata)
            self.transcriptions.append(('Median',intervals,labels))
            
            rp_chromadata,self.ground_truth,intervals,labels = load_trackdata(self.filepaths[0],
                    self.dropdown_id.value,self.dropdown_dataset.value,
                    "majmin_intervals","majmin_labels")
            self.chromadata.append(rp_chromadata)
            self.transcriptions.append(('RP',intervals,labels))

            self.updateControls(self.chromadata[0][0][-1], self.chromadata[0][1])
            return
        elif self.dropdown_resultfile.label == "distance metrics":
            self.transcriptions = []
            self.chromadata, self.ground_truth, intervals_corr, labels_corr = load_trackdata(self.filepaths[0],
                    self.dropdown_id.value,self.dropdown_dataset.value,
                    "sevenths_intervals_correlation","sevenths_labels_correlation")
            self.transcriptions.append(('Correlation',intervals_corr,labels_corr))
            _,_,intervals,labels = load_trackdata(self.filepaths[0],
                    self.dropdown_id.value,self.dropdown_dataset.value,
                    "sevenths_intervals_inner_product","sevenths_labels_inner_product")
            self.transcriptions.append(('inner_product',intervals,labels))
            self.updateControls(self.chromadata[0][-1], self.chromadata[1])
            return
        elif self.dropdown_resultfile.label == "pitchspace":
            self.transcriptions = []
            _, _,intervals,labels = load_trackdata(self.filepaths[0],
                    self.dropdown_id.value,self.dropdown_dataset.value,
                    "intervals_templates","labels_templates")
            self.transcriptions.append(('Templates',intervals,labels))
            self.chromadata, self.ground_truth,intervals,labels = load_trackdata(self.filepaths[0],
                    self.dropdown_id.value,self.dropdown_dataset.value,
                    "intervals_cpss","labels_cpss")
            self.transcriptions.append(('CPSS',intervals,labels))
            self.updateControls(self.chromadata[0][-1], self.chromadata[1])    
            _,_, intervals_corr, labels_corr = load_trackdata(self.filepaths[0],
                    self.dropdown_id.value,self.dropdown_dataset.value,
                    "intervals_stableregions_cpss","labels_stableregions_cpss")
            self.transcriptions.append(('stable regions',intervals_corr,labels_corr))
        elif self.dropdown_resultfile.label == "deepchroma":
            self.transcriptions = []
            self.chromadata = []

            # load CRP chroma
            crp_chroma, _,intervals,labels = load_trackdata(self.filepaths[0],
                    self.dropdown_id.value,self.dropdown_dataset.value,
                    "intervals_templates","labels_templates")
            self.chromadata.append(crp_chroma)

            # load Template transcription with deepchroma
            _, _,intervals,labels = load_trackdata(self.filepaths[1],
                    self.dropdown_id.value,self.dropdown_dataset.value,
                    "est_templates_intervals","est_templates_labels")
            self.transcriptions.append(('Templates',intervals,labels))

            # load cpss transcription with deepchroma
            _, _,intervals,labels = load_trackdata(self.filepaths[1],
                    self.dropdown_id.value,self.dropdown_dataset.value,
                    "est_cpss_intervals","est_cpss_labels")
            self.transcriptions.append(('CPSS',intervals,labels))

            # load madmom transcription
            deepchroma, self.ground_truth,intervals,labels = load_trackdata(self.filepaths[2],
                    self.dropdown_id.value,self.dropdown_dataset.value,
                    "majmin_intervals","majmin_labels")
            self.transcriptions.append(('DCP',intervals,labels))
            self.chromadata.append(deepchroma)

            # update controls
            self.updateControls(self.chromadata[0][0][-1], self.chromadata[0][1])

    def updateControls(self,t_max,cqt):
        """update T-slider and CQT button"""
        self.t_start_slider.max = t_max - self.delta_t.value
        self.t_start_slider.disabled = False
        self.delta_t.disabled = False

    def update_dataset(self,*args):
        """callback function for dropdown_dataset"""
        self.update_dropdown_id_options() # load new track_ids when dataset changed

    def update_dropdown_id_options(self,*args):
        self.dropdown_id.value = None
        self.dropdown_id.options = [(x.name,x.track_id) for x in self.tracklist if x.dset == self.dropdown_dataset.value]
        self.update_selected_track_id()

    def update_selected_track_id(self,*args):
        if self.dropdown_id.value is not None:
            self.load_track()
            self.text.value = self.dropdown_id.value
