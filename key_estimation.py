import utilities
import features
import librosa
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import mir_eval
import pitchspace

# figure related parameters
SAVE = False
name = "key_estimation_pc_energy.png"
basepath = "/home/max/ET-TI/Masterarbeit/latex/figures"

start = 0
stop = 10
path = "/home/max/ET-TI/Masterarbeit/mirdata/beatles/"

titles = ["06_-_Rubber_Soul/11_-_In_My_Life", "12_-_Let_It_Be/06_-_Let_It_Be","10CD1_-_The_Beatles/CD1_-_17_-_Julia"]
# title = "06_-_Rubber_Soul/11_-_In_My_Life"
title = "12_-_Let_It_Be/06_-_Let_It_Be"
title = "10CD1_-_The_Beatles/CD1_-_17_-_Julia"
audiopath = f"{path}/audio/{title}.wav"

# the exception handling is necessary due to a weird bug, if i try to load it the first time it doesnt work!
# maybe because of the f-string i used?
try:
    y,sr = librosa.load(audiopath,mono=True,offset=start,duration=stop-start,sr=22050)
except TypeError as e: 
    print(e)
    y,sr = librosa.load(audiopath,mono=True,offset=start,duration=stop-start,sr=22050)

y = y / np.max(y)
target = mir_eval.io.load_labeled_intervals(path+"/annotations/chordlab/The Beatles/"+title+".lab",' ','#')
t, chroma = features.crpChroma(y,nCRP=22)

### BASIS ALGORITHMUS / PARAMETERSTUDIE ENERGIESCHWELLE, WINKELGEWICHTUNG
if False:
    fig,ax = plt.subplots(6,1,height_ratios=(1,3,3,3,3,3),figsize=(9,11))
    utilities.plotChordAnnotations(ax[0],target,(start,stop))

    utilities.plotChromagram(ax[1],t,chroma)

    pc_energy = pitchspace.getPitchClassEnergyProfile(chroma,threshold=0.75,angle_weight=0.75)
    utilities.plotChromagram(ax[2],t,pc_energy)
    ax[2].text(0,12,f"thresh={0.75},angle={0.7}",fontsize=11)

    pc_energy = pitchspace.getPitchClassEnergyProfile(chroma,threshold=0.3,angle_weight=0.75)
    utilities.plotChromagram(ax[3],t,pc_energy)
    ax[3].text(0,12,f"thresh={0.6},angle={0.7}",fontsize=11)

    pc_energy = pitchspace.getPitchClassEnergyProfile(chroma,threshold=0.3,angle_weight=0.5)
    utilities.plotChromagram(ax[4],t,pc_energy)
    ax[4].text(0,12,f"thresh={0.5},angle={0.5}",fontsize=11)

    pc_energy = pitchspace.getPitchClassEnergyProfile(chroma,threshold=0.8,angle_weight=0.9)
    utilities.plotChromagram(ax[5],t,pc_energy)
    ax[5].text(0,12,f"thresh={0.8},angle={0.9}",fontsize=11)

    for i in [1,2,3,4]:
        ax[i].set_xlabel("")
        ax[i].set_xticklabels([])
    if SAVE:
        plt.savefig(basepath+"/"+name)
    plt.show()

if False: # evaluate Chord Prototypes instead of chroma
    triads,labels_triads = utilities.chordTemplates("triads")
    tetrads, labels_tetrads = utilities.chordTemplates("tetrads")
    templates = np.concatenate((triads,tetrads),axis=0)
    t_templates = np.linspace(start,stop,templates.shape[0],endpoint=True)
    labels = labels_triads + labels_tetrads

    # overwrite chroma/annotations with prototype information
    target = utilities.createChordIntervals(t_templates,labels)
    t = t_templates
    chroma = templates

def estimateKeys(chroma,ic_threshold=0.01):
    # other approach to calculate correlation with major key profiles (krumhansl)
    templates = np.zeros((12,12),dtype=float)
    key_profile = np.array([5, 2, 3.5, 2, 4.5, 4, 2, 4.5, 2, 3.5, 1.5, 4])/12 # (12,) arbitrary normation for plots
    # key_profile = key_profile / np.sum(key_profile)
    for i in range(12):
        templates[i,:] = np.roll(key_profile,i)
    correlation = np.matmul(templates,chroma.T).T # chroma shape (t,12)

    # computation of interval categories
    key_profile = [0,2,4,5,7,9,11]
    key_template = np.zeros_like(chroma)
    key_template[:, key_profile] = 1.0
    # precompute interval_categories for all keys
    interval_categories = np.zeros((12,chroma.shape[0],6))
    ic_energy = np.zeros_like(chroma)
    for pitch_class in range(12):
        key_related_chroma = np.multiply(chroma,np.roll(key_template,pitch_class))
        interval_categories[pitch_class,:,:] = features.intervalCategories(key_related_chroma)
        ic_energy[:,pitch_class] = np.sum(interval_categories[pitch_class,:,2:5], axis=1)  # sum energy of m3/M6,M3/m6,P4/P5  

    # we can savely discard some keys with low correlation (6 or 8?)
    correlation_energy = np.square(correlation)
    key_candidates = np.argsort(correlation_energy,axis=1)[:,-3:]

    keys = []
    for i in range(key_candidates.shape[0]): # for all timesteps
        candidates = []
        for n in range(3): # 3 candidates
            ic = interval_categories[key_candidates[i,n],i,:]
            if ic[2] > ic_threshold and ic[3] > ic_threshold and ic[4] > ic_threshold:
               candidates.append((key_candidates[i,n],correlation_energy[i,key_candidates[i,n]]+ic_energy[i,n]))
        keys.append(candidates) 
    return keys

    # # define gridspec and axes
    # fig = plt.figure(figsize=(9,11))
    # gs = matplotlib.gridspec.GridSpec(5, 3,width_ratios=(40,1,5),height_ratios=(1,6,6,6,6), hspace=0.2,wspace=0.05)
    # ax00 = fig.add_subplot(gs[0,0])
    # # ax01 = fig.add_subplot(gs[0,1])
    # ax10 = fig.add_subplot(gs[1,0])
    # ax11 = fig.add_subplot(gs[1,1])
    # ax20 = fig.add_subplot(gs[2,0])
    # ax21 = fig.add_subplot(gs[2,1])
    # ax30 = fig.add_subplot(gs[3,0])
    # ax31 = fig.add_subplot(gs[3,1])
    # ax40 = fig.add_subplot(gs[4,0])
    # ax41 = fig.add_subplot(gs[4,1])
    # utilities.plotChordAnnotations(ax00,target,(start,stop))
    # img = utilities.plotChromagram(ax10,t,chroma)
    # fig.colorbar(img,cax=ax11)
    # img = utilities.plotChromagram(ax20,t,correlation_energy,vmax=0.5)
    # fig.colorbar(img,cax=ax21)
    # ax20.text(0,12,"Correlation")
    # img = utilities.plotChromagram(ax30,t,ic_energy,vmax=0.5)
    # fig.colorbar(img,cax=ax31)
    # ax30.text(0,12,"IC-Energy")
    # img = utilities.plotChromagram(ax40,t,correlation_energy+ic_energy,vmax=0.5)
    # ax40.text(0,12,"Sum")
    # fig.colorbar(img,cax=ax41)
    # ax10.plot(t,key_candidates,'ok',markersize=1)
    # plt.show()
    
keys = estimateKeys(chroma,0.005)


# PLOT IC ENERGY FOR IN MY LIFE
fig = plt.figure(figsize=(9,11))
gs = matplotlib.gridspec.GridSpec(5,1,width_ratios=(10,),height_ratios=(1,6,3,3,3), hspace=0.4,wspace=0.05)
ax0 = fig.add_subplot(gs[0,0])
ax1 = fig.add_subplot(gs[1,0])
utilities.plotChordAnnotations(ax0,target,(start,stop))
utilities.plotChromagram(ax1,t,chroma)

for i,candidates in enumerate(keys):
    try:
        ax1.plot(t[i],candidates[0],'ok',markersize=3)
        ax1.plot(t[i],candidates[1],'og',markersize=3)
        ax1.plot(t[i],candidates[2],'xb',markersize=3)
    except IndexError:
        pass

plt.show()
quit()
# PLOT RESULTS KEY ESTIMATION FOR 3 SONGS
fig = plt.figure(figsize=(9,11))
gs = matplotlib.gridspec.GridSpec(6,3,width_ratios=(40,1,6),height_ratios=(1,6,1,6,1,6), hspace=0.4,wspace=0.05)
axes = []
ax00 = fig.add_subplot(gs[0,0])
ax01 = fig.add_subplot(gs[0,1:])
ax10 = fig.add_subplot(gs[1,0])
ax11 = fig.add_subplot(gs[1,1])
axes.append([ax00,ax01,ax10,ax11])

ax20 = fig.add_subplot(gs[2,0])
ax21 = fig.add_subplot(gs[2,1:])
ax30 = fig.add_subplot(gs[3,0])
ax31 = fig.add_subplot(gs[3,1])
axes.append([ax20,ax21,ax30,ax31])

ax40 = fig.add_subplot(gs[4,0])
ax41 = fig.add_subplot(gs[4,1:])
ax50 = fig.add_subplot(gs[5,0])
ax51 = fig.add_subplot(gs[5,1])
axes.append([ax40,ax41,ax50,ax51])

for i,title in enumerate(titles):
    audiopath = f"{path}/audio/{title}.wav"

    # the exception handling is necessary due to a weird bug, if i try to load it the first time it doesnt work!
    # maybe because of the f-string i used?
    try:
        y,sr = librosa.load(audiopath,mono=True,offset=start,duration=stop-start,sr=22050)
    except TypeError as e: 
        print(e)
        y,sr = librosa.load(audiopath,mono=True,offset=start,duration=stop-start,sr=22050)

    y = y / np.max(y)
    target = mir_eval.io.load_labeled_intervals(path+"/annotations/chordlab/The Beatles/"+title+".lab",' ','#')
    t, chroma = features.crpChroma(y,nCRP=22)
    key_candidates = estimateKeys(chroma)

    utilities.plotChordAnnotations(axes[i][0],target,(start,stop))
    axes[i][2].text(0,12,title)
    axes[i][1].axis("off")
    img = utilities.plotChromagram(axes[i][2],t,chroma)
    fig.colorbar(img,cax=axes[i][3])
    axes[i][2].plot(t,key_candidates,'ok',markersize=1)

plt.show()
