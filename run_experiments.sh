#!/bin/bash

# parameters
transition_probs=(0.2 0.3 0.4)
eval_schemes=('majmin' 'sevenths')


# Loop through combinations of filter lengths and transition probabilities
for eval_scheme in "${eval_schemes[@]}"; do
  for transition_prob in "${transition_probs[@]}"; do
    dynamic_filename="${eval_scheme}_median_7_prob_$(echo $transition_prob | sed 's/\.//g')"
    python transcribe.py "$dynamic_filename" --prefilter median --prefilter_length 7 --transition_prob "$transition_prob" --eval_scheme "$eval_scheme"
  done
done
