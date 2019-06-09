"""Chop midi files into segments containing a given number of bars."""

import argparse
import collections
import logging
import math
import pickle
import re

import pretty_midi


MIN_DURATION = 1e-5  # Most likely below MIDI resolution
LOGGER = logging.getLogger('ismir2019_cifka')


def chop_midi(files, bars_per_segment, instrument_re=None, programs=None, drums=None,
              min_notes_per_segment=1, include_segment_id=False, force_tempo=None, skip_bars=0):
    if isinstance(bars_per_segment, int):
        bars_per_segment = [bars_per_segment]
    bars_per_segment = list(bars_per_segment)

    for file in files:
        if isinstance(file, str):
            file_id = file
        else:
            file_id = file.name
        midi = pretty_midi.PrettyMIDI(file)

        if force_tempo is not None:
            normalize_tempo(midi, force_tempo)

        instruments = midi.instruments
        if instrument_re is not None:
            instruments = [i for i in instruments if re.search(instrument_re, i.name)]
        if programs is not None:
            instruments = [i for i in instruments if i.program + 1 in programs]
        if drums is not None:
            # If True, match only drums; if False, match only non-drums.
            instruments = [i for i in instruments if i.is_drum is drums]

        if not instruments:
            LOGGER.warning(f'Could not match any track in file {file_id}; skipping file')
            continue
        all_notes = [n for i in instruments for n in i.notes]
        all_notes.sort(key=lambda n: n.start)

        def is_overlapping(note, start, end):
            """Check whether the given note overlaps with the given interval."""
            # If a note's start "isclose" to a segment boundary, we want to include it
            # in the following segment only.
            # Note that if the note is extremely short, it might end before the segment starts!
            return ((note.end > start or math.isclose(note.start, start)) and
                    note.start < end and not math.isclose(note.start, end))

        downbeats = midi.get_downbeats()[skip_bars:]
        for bps in bars_per_segment:
            note_queue = collections.deque(all_notes)
            notes = []  # notes in the current segment
            for i in range(0, len(downbeats), bps):
                start = downbeats[i]
                end = downbeats[i + bps] if i + bps < len(downbeats) else midi.get_end_time()
                if math.isclose(start, end):
                    continue

                # Filter the notes from the previous segment to keep those that overlap with the
                # current one.
                notes[:] = (n for n in notes if is_overlapping(n, start, end))

                # Add new overlapping notes. note_queue is sorted by onset time, so we can stop
                # after the first note which is outside the segment.
                while note_queue and is_overlapping(note_queue[0], start, end):
                    notes.append(note_queue.popleft())

                # Clip the notes to the segment.
                notes_clipped = [
                    pretty_midi.Note(
                        start=max(0., n.start - start),
                        end=max(0., min(n.end, end) - start),
                        pitch=n.pitch,
                        velocity=n.velocity)
                    for n in notes
                ]
                # Remove extremely short notes that could have been created by clipping.
                notes_clipped = [n for n in notes_clipped if n.end-n.start >= MIN_DURATION]

                if len(notes_clipped) < min_notes_per_segment:
                    continue

                if include_segment_id:
                    yield ((file_id, i, i + bps), notes_clipped)
                else:
                    yield notes_clipped


def normalize_tempo(midi, new_tempo=60):
    if math.isclose(midi.get_end_time(), 0.):
        return

    tempo_change_times, tempi = midi.get_tempo_changes()
    original_times = list(tempo_change_times) + [midi.get_end_time()]
    new_times = [original_times[0]]

    # Iterate through all the segments between the tempo changes.
    # Compute a new duration for each of them.
    for start, end, tempo in zip(original_times[:-1], original_times[1:], tempi):
        time = (end - start) * tempo / new_tempo
        new_times.append(new_times[-1] + time)

    midi.adjust_times(original_times, new_times)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_files', nargs='+', metavar='FILE', help='input MIDI files')
    parser.add_argument('output_file', type=argparse.FileType('wb'), metavar='OUTPUTFILE',
                        help='output pickle file')

    parser.add_argument('-i', '--instrument-re', type=str, default='.*', metavar='REGEXP',
                        help='select only tracks matching this regular expression')
    parser.add_argument('-p', '--program', type=lambda l: [int(x) for x in l.split(',')],
                        default=None,
                        help='select only the given MIDI program numbers (comma-separated list)')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--drums', action='store_true', help='select only drum tracks')
    group.add_argument('--no-drums', action='store_false', dest='drums', help='exclude drum tracks')
    group.set_defaults(drums=None)

    parser.add_argument('-b', '--bars-per-segment', metavar='COUNT',
                        type=lambda l: [int(x) for x in l.split(',')],
                        default=[8], help='the desired segment length in bars')
    parser.add_argument('-n', '--min-notes-per-segment', type=int, default=1, metavar='COUNT',
                        help='drop segments with fewer notes than the given value; default: 1')
    parser.add_argument('-t', '--force-tempo', type=float, default=None, metavar='BPM',
                        help='normalize the tempo to the given value')
    parser.add_argument('--skip-bars', type=int, default=0, metavar='COUNT',
                        help='skip the given number of initial bars')
    parser.add_argument('--include-segment-id', action='store_true',
                        help='save each segment as a tuple containing the segment ID and a list of '
                             'notes; without this option, only the note list is stored')
    args = parser.parse_args()

    output = list(chop_midi(files=args.input_files,
                            bars_per_segment=args.bars_per_segment,
                            instrument_re=args.instrument_re,
                            programs=args.program,
                            drums=args.drums,
                            min_notes_per_segment=args.min_notes_per_segment,
                            include_segment_id=args.include_segment_id,
                            force_tempo=args.force_tempo,
                            skip_bars=args.skip_bars))

    pickle.dump(output, args.output_file)


if __name__ == '__main__':
    main()
