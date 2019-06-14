#!/bin/bash

trap "exit 1" INT

style_list=../data/parallel/styles
style_profile_dir=../data/parallel/style_profiles
bodhidharma_dir=../data/bodhidharma/04_chopped
out_dir=out_bodh

set -x

mkdir -p {all2bass,all2piano,bass2bass_roll2seq,bass2bass_seq2seq,piano2piano}/"$out_dir"
mkdir -p results


# Translate the test set to all styles using all models.

cat "$style_list" | while read -r style; do
  for dir in all2bass all2piano; do
    python -m ismir2019_cifka.models.roll2seq_style --logdir "$dir" run \
      "$bodhidharma_dir/all_except_drums.pickle" "$dir/$out_dir/$style.pickle" "$style"
  done
  for model in roll2seq seq2seq; do
    dir=bass2bass_$model
    python -m ismir2019_cifka.models.${model}_style --logdir "$dir" run \
      "$bodhidharma_dir/bass.pickle" "$dir/$out_dir/$style.pickle" "$style"
  done
  dir=piano2piano
  python -m ismir2019_cifka.models.roll2seq_style --logdir "$dir" run \
    "$bodhidharma_dir/piano_organ.pickle" "$dir/$out_dir/$style.pickle" "$style"
done


# Compute the metrics.

python -m ismir2019_cifka.evaluate \
  --models all2bass bass2bass_roll2seq bass2bass_seq2seq \
  --data-prefix "$out_dir/" \
  --source "$bodhidharma_dir/bass.pickle" \
  --style-profile-dir "$style_profile_dir/Bass" \
  --style-list "$style_list" \
  --output results/test_bass.pickle

python -m ismir2019_cifka.evaluate \
  --models all2piano piano2piano \
  --data-prefix "$out_dir/" \
  --source "$bodhidharma_dir/piano_organ.pickle" \
  --style-profile-dir "$style_profile_dir/Piano" \
  --style-list "$style_list" \
  --output results/test_piano.pickle
