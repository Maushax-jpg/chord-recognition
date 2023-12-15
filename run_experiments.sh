#!/bin/bash

# Loop through combinations of source separation techniques
python transcribe.py "results/none_median_7_prob_025" --prefilter median --prefilter_length 7 --transition_prob "0.25"
source_separation=("vocals" "drums" "both")
for source_sep in "${source_separation[@]}"; do
    dynamic_filename="results/${source_sep}_median_7_prob_025"
    python transcribe.py "$dynamic_filename" --prefilter median --prefilter_length 7 --transition_prob "0.25" --source_separation "${source_sep}"
done
