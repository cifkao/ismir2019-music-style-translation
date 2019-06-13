#!/bin/bash
# Run this script to prepare the data in the current directory.
# Point the src_dir variable below to the directory containing
# your MIDI files before running the script.
# See README.md for more information.

shopt -s extglob
set -o pipefail
trap "exit 1" INT


### OPTIONS ###
src_dir=01_src
n_shards=60


function die { [[ $# > 0 ]] || set -- Failed.; echo; echo >&2 "$@"; exit 1; }
function log { echo >&2 "$@"; }
function log_progress { echo -en "\r\033[2K$@ "; }


log "Preparing data in $PWD"

[[ -d $src_dir ]] || log "Warning: $src_dir does not exist"

# Filter the files to have 4/4 time only
dir=02_filtered
[[ -d $src_dir ]] && mkdir "$dir" && {
  log "Found $(find -L "$src_dir" -type f -name '*.mid' | wc -l) files in $src_dir"
  ignored_count=$(find -L "$src_dir" -type f -not -name '*.mid' | wc -l)
  if [[ $ignored_count -gt 0 ]]; then
    log "Ignoring $ignored_count non-MIDI files:"
    find -L "$src_dir" -type f -not -name '*.mid' >&2
  fi

  python -m ismir2019_cifka.data.filter_4beats "$src_dir"/*.mid | while read f; do
    log_progress "$(basename "$f")"
    ln "$f" "$dir/$(basename "$f")" || die
  done || die
  log
  log "Linked $(find "$dir" -name '*.mid' | wc -l) files to $dir"
}

# Distribute the files into shards
dir=03_sharded
[[ -d 02_filtered ]] && mkdir "$dir" && {
  if [[ -e shards.tsv ]] ; then
    cp shards.tsv "$dir"
  else
    log "shards.tsv not provided, sharding randomly"
    find 02_filtered -name '*.mid' -printf '%f\n' | sed -r 's/\.[^.]+\.[0-9]+\.mid$//' | sort -u >"$dir/songs"
    seq -w 0 $((n_shards-1)) | shuf -r -n $(wc -l <"$dir/songs") | paste "$dir/songs" - >"$dir/shards.tsv"
  fi

  while IFS=$'\t' read -r name shard; do
    mkdir -p "$dir/$shard"
    log_progress $shard/$name
    ln -t "$dir/$shard" "02_filtered/$name".!(*.*).[0-9][0-9].mid || die
  done <"$dir/shards.tsv"
  log
  log "Linked $(find "$dir" -name '*.mid' | wc -l) files to $dir"
}

# Chop the files into 8-bar segments, store them as note lists
dir=04_chopped
[[ -d 03_sharded ]] && mkdir "$dir" && {
  for shard in $(seq -w 0 $((n_shards-1))); do
    for instr in Drums Bass Piano; do
      mkdir -p "$dir/$instr"
      log_progress $instr/$shard
      python -m ismir2019_cifka.data.chop_midi \
          --instrument-re "^BB $instr$" \
          --skip-bars 2 \
          --bars-per-segment 8 \
          --min-notes-per-segment 1 \
          --force-tempo 60 \
          --include-segment-id \
          03_sharded/$shard/*.mid "$dir/$instr/$shard.pickle" \
          || die

      # The segment ID contains the entire file path. We need to adjust it to have only the
      # song name and the style (i.e. the filename without the .mid extension and the number).
      python -m ismir2019_cifka.data.adjust_segment_ids --strip-dirs --strip-exts 2 "$dir/$instr/$shard.pickle" || die
    done

    instr=all_except_drums
    mkdir -p "$dir/$instr"
    log_progress $instr/$shard
    python -m ismir2019_cifka.data.chop_midi \
        --instrument-re "^BB (Bass|Piano|Guitar|Strings)$" \
        --skip-bars 2 \
        --bars-per-segment 8 \
        --min-notes-per-segment 1 \
        --force-tempo 60 \
        --include-segment-id \
        03_sharded/$shard/*.mid "$dir/$instr/$shard.pickle" \
        || die

    python -m ismir2019_cifka.data.adjust_segment_ids --strip-dirs --strip-exts 2 "$dir/$instr/$shard.pickle" || die
  done
  log
}

[[ -d "$dir" ]] || die

log Done.

exit 0
