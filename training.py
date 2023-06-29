import dataloader
import numpy as np
import mir_eval
import pandas as pd
import matplotlib.pyplot as plt
import pickle

if __name__ == "__main__":
    for split in range(1,8):
        dataset = dataloader.MIRDataset("beatles", use_deep_chroma=True, align_chroma=False, split_nr=split)
        
        df_data = []  # List to store data for DataFrame
        
        for id in dataset.getTrackList():
            audio, features, target = dataset[id]
            chroma = features["chroma"]
            t_chroma = features["time"]
            intervals, labels = target
            
            for interval, label in zip(intervals, labels):
                try:
                    index_start = int(np.argwhere(t_chroma >= interval[0])[0])
                    index_stop = int(np.argwhere(t_chroma >= interval[1])[0])
                except IndexError:
                    continue
                chroma_slice = chroma[index_start:index_stop, :]
                for chromavector in chroma_slice:
                    df_data.append(np.append(chromavector, label))
        # Create the DataFrame with column names
        columns = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B","label"]
        df = pd.DataFrame(df_data, columns=columns)
        # Save the DataFrame using pickle
        with open(f"/home/max/ET-TI/Masterarbeit/datasets/beatles/chroma/{split}_deep_chroma.pkl", "wb") as f:
            pickle.dump(df, f)