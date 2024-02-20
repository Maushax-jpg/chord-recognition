import numpy as np
import mir_eval
import h5py
import os
import pitchspace
from tqdm import tqdm
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser(prog='create a circular pitch space model', description='creates labeled chromadata from existing chromagrams')
    parser.add_argument('alphabet', default="majmin", choices=["majmin","sevenths"])
    args = parser.parse_args()

    # create a default path 
    script_directory = os.path.dirname(os.path.abspath(__file__))

    for split in tqdm(range(9),desc="split"): 
        chromadata = {}
        filepath = os.path.join(script_directory, "models","chromadata",f"chromadata_{split}.hdf5")
        with  h5py.File(filepath,"r") as file:
            for group in file:
                if group == "N":
                    continue
                root,qual,scale_deg,bass = mir_eval.chord.split(group)
                if np.array(file[group].get(group)).size != 0 :
                    chromadata[qual] = np.copy(file[group].get(group))

        # offset in template array (see utils.createChordTemplates)
        offsets = {"maj":0,"min":12,"maj7":24,"7":36,"min7":48} 
        shift = [0,2,4,5,7,9]
        keys = {}
        for key_index in range(12):
            chordmodels = []
            # create the triads for all major keys  
            for i,qual in tqdm(zip(shift,["maj","min","min","maj","maj","min"]),desc=f"key {key_index}",total=7):
                chroma = np.roll(chromadata[qual],key_index + i,axis=0)
                # transform onto pitchspace 
                F,FR,TR,_ = pitchspace.computeCPSSfeatures(chroma)
                # rearange features in a tuple
                x = (
                    F[0,:], F[1,:],
                    FR[2 * key_index, :], FR[2*key_index+1, :],
                    TR[2 * key_index, :], TR[2*key_index+1, :]
                )
                chord_index = (key_index + i) % 12 + offsets[qual]
                chordmodels.append(pitchspace.ChordModel(chord_index, x))

            if args.alphabet == "sevenths":
                # create the tetrads for all major keys  
                for i,qual in tqdm(zip(shift,["maj7","min7","min7","maj7","7","min7"]),desc="tetrads",total=7):
                    chroma = np.roll(chromadata[qual],key_index + i,axis=0)
                    # transform onto pitchspace 
                    F,FR,TR,_ = pitchspace.computeCPSSfeatures(chroma)
                    # rearange features in a tuple
                    x = (
                        F[0,:], F[1,:],
                        FR[2 * key_index, :], FR[2*key_index+1, :],
                        TR[2 * key_index, :], TR[2*key_index+1, :]
                    )
                    chord_index = (key_index + i) % 12 + offsets[qual]
                    chordmodels.append(pitchspace.ChordModel(chord_index, x))
            keys[key_index] = chordmodels
        outputpath = os.path.join(script_directory, "models","cpss",f"cpss_{args.alphabet}_{split}.npy")
        np.save(outputpath,keys)
