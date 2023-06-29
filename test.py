import pickle
import matplotlib.pyplot as plt

with open("/home/max/ET-TI/Masterarbeit/datasets/beatles/chroma/1_deep_chroma.pkl", "rb") as f:
    df = pickle.load(f)

# Group the DataFrame by label
grouped = df.groupby('label')

for label, group in grouped:
    chromavectors = group.drop('label', axis=1).values

