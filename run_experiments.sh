#!/bin/bash

# Loop through combinations of source separation techniques
source_separation=("none" "vocals" "drums" "both")
for source_sep in "${source_separation[@]}"; do
    dynamic_filename="results/source_separation_${source_sep}"
    python source_separation_exp.py "$dynamic_filename" --source_separation "${source_sep}"
done
