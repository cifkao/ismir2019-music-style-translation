#!/bin/bash
# Run this script to prepare the Bodhidharma data in the current directory.
# Will download the dataset first. See README.md for more information.

shopt -s extglob
set -o pipefail
trap "exit 1" INT

src_dir=MIDI_Files

function die { [[ $# > 0 ]] || set -- Failed.; echo; echo >&2 "$@"; exit 1; }
function log { echo >&2 "$@"; }
function log_progress { echo -en "\r\033[2K$@ "; }


if [[ ! -e "$src_dir" ]]; then
  wget http://www.music.mcgill.ca/~cmckay/protected/Bodhidharma_MIDI.zip || die
  unzip -n Bodhidharma_MIDI.zip || die
  rm -f Bodhidharma_MIDI.zip
fi
[[ -e "$src_dir" ]] || die "$src_dir does not exist"

# Fix the time signatures and filenames
dir=01_fixed
mkdir "$dir" && {
  log "Found $(find -L "$src_dir" -type f | wc -l) files in $src_dir"
  find "$src_dir" | grep -Ei '\.mid$' | while read -r f; do
    fname="$(basename "$f" | sed -r 's/\.mid/.mid/i')"
    log_progress "$fname"
    python -m ismir2019_cifka.data.fix_time_signatures "$f" "$dir/$fname"
  done || die
  log
  log "Created $(find "$dir" -name '*.mid' | wc -l) files in $dir"
}

# Filter the files to have 4/4 time only
dir=02_filtered
mkdir "$dir" && {
  python -m ismir2019_cifka.data.filter_4beats 01_fixed/*.mid | while read -r f; do
    log_progress "$(basename "$f")"
    ln "$f" "$dir/$(basename "$f")" || die
  done || die
  log
  log "Linked $(find "$dir" -name '*.mid' | wc -l) files to $dir"
}

# Separate the instrument tracks
dir=03_separated
mkdir "$dir" && {
  for instr in drums bass piano piano_organ all_except_drums; do
    mkdir -p "$dir/$instr"
  done

  for f in 02_filtered/*.mid; do
    log_progress "$f"

    instr=drums
    python -m ismir2019_cifka.data.filter_midi_messages \
      --channel 10 \
      "$f" "$dir/$instr/$(basename "$f")" \
      || die

    instr=bass
    python -m ismir2019_cifka.data.filter_midi_messages \
      --except-channel 10 \
      --program 33,34,35,36,37,38,39,40 \
      "$f" "$dir/$instr/$(basename "$f")" \
      || die

    instr=piano
    python -m ismir2019_cifka.data.filter_midi_messages \
      --except-channel 10 \
      --program 1,2,3,4,5,6,7,8 \
      "$f" "$dir/$instr/$(basename "$f")" \
      || die

    instr=piano_organ
    python -m ismir2019_cifka.data.filter_midi_messages \
      --except-channel 10 \
      --program 1,2,3,4,5,6,7,8,17,18,19,20,21,22,23,24 \
      "$f" "$dir/$instr/$(basename "$f")" \
      || die

    instr=all_except_drums
    python -m ismir2019_cifka.data.filter_midi_messages \
      --except-channel 10 \
      "$f" "$dir/$instr/$(basename "$f")" \
      || die
  done
  log
}

# Categorize the files by genre
dir=04_by_genre
mkdir "$dir" && {
  for instr in drums bass piano piano_organ all_except_drums; do
    cat recordings_key.tsv | while IFS=$'\t' read -r fname song artist genre; do
      fname="$(echo "$fname" | sed -r 's/\.mid/.mid/i')"
      mkdir -p "$dir/$instr/$genre" "$dir/$instr/__all__"
      log_progress "$instr/$genre/$fname"
      if [[ -e "03_separated/$instr/$fname" ]]; then
        ln "03_separated/$instr/$fname" "$dir/$instr/$genre/$fname"
        ln "03_separated/$instr/$fname" "$dir/$instr/__all__/$fname"
      fi
    done
    log
    log "Linked $(find "$dir/$instr/__all__" -name '*.mid' | wc -l) files to $dir/$instr/__all__"
  done
}

# Chop the files into 8-bar segments, store them as note lists
dir=05_chopped
mkdir "$dir" && {
  { echo __all__; cat recordings_key.tsv | cut -f 4 | sort -u; } | while read -r genre; do
    for instr in drums bass piano piano_organ all_except_drums; do
      mkdir -p "$dir/$instr"
      log_progress $instr/$genre

      if find "04_by_genre/$instr/$genre" -name '*.mid' -print -quit | grep -q .; then
      python -m ismir2019_cifka.data.chop_midi \
          --bars-per-segment 8 \
          --min-notes-per-segment 1 \
          --force-tempo 60 \
          --include-segment-id \
	        "04_by_genre/$instr/$genre"/*.mid "$dir/$instr/$genre.pickle" \
	        || die

        python -m ismir2019_cifka.data.adjust_segment_ids --strip-dirs --strip-exts 1 "$dir/$instr/$genre.pickle" || die
      fi
    done
  done
  log
}

[[ -d "$dir" ]] || die

log Done.

exit 0
