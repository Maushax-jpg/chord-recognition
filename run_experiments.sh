#!/bin/bash

# Loop through combinations of source separation techniques
# python transcribe.py "results/none_median_7_prob_025" --prefilter median --prefilter_length 7 --transition_prob "0.25"
# source_separation=("vocals" "drums" "both")
# for source_sep in "${source_separation[@]}"; do
#     dynamic_filename="results/${source_sep}_median_7_prob_025"
#     python transcribe.py "$dynamic_filename" --prefilter median --prefilter_length 7 --transition_prob "0.25" --source_separation "${source_sep}"
# done

probabilities=(0.05 0.1 0.15 0.2 0.25 0.3)

for prob in "${probabilities[@]}"; do
     dynamic_filename="results/none_median_7_prob_${prob}"
     python transcribe.py "$dynamic_filename" --prefilter median --prefilter_length 7 --transition_prob "${prob}" 
done
