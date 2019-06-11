"""Turn pickled sequences of notes into MIDI files."""

import argparse
import collections
import os
import pickle

import pretty_midi


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.set_defaults(func=main)
    parser.add_argument('input_file', type=argparse.FileType('rb'), metavar='FILE')
    parser.add_argument('output_dir', type=str, metavar='OUTPUTDIR')
    parser.add_argument('-i', '--instrument', type=str, default='')
    parser.add_argument('--drums', action='store_true')
    parser.add_argument('-p', '--program', type=int)
    parser.add_argument('--stretch', type=str)
    parser.add_argument('--tempo', type=float)
    parser.add_argument('--time-signature', type=lambda s: tuple(int(x) for x in s.split('/')),
                        default=(4, 4))
    parser.add_argument('--resolution', type=int, default=480)
    parser.add_argument('--range', type=lambda r: [None if x == '' else int(x)
                                                   for x in r.split(':')])
    parser.add_argument('--group-by-name', action='store_true')
    parser.add_argument('--time-unit', type=float)
    args = parser.parse_args()

    if args.program is None:
        if args.instrument:
            args.program = pretty_midi.instrument_name_to_program(args.instrument)
        else:
            args.program = 0

    tempo = 60.
    if args.stretch:
        # Calculate the time stretch ratio
        if ':' in args.stretch:
            a, b = args.stretch.split(':')
            args.stretch = float(a) / float(b)
            tempo = float(b)
        else:
            args.stretch = float(args.stretch)
            tempo = tempo / args.stretch
    if args.tempo:
        tempo = args.tempo

    data = pickle.load(args.input_file)
    fill_length = len(str(len(data) - 1))

    if args.range:
        data = data[slice(*args.range)]

    if args.group_by_name:
        grouped = collections.defaultdict(list)
        for (name, start, end), notes in data:
            start, end = start * args.time_unit, end * args.time_unit
            grouped[name].extend(_shift_and_clip(note, start, end)
                                 for note in notes
                                 if note.start + start < end and note.end > 0.)
        data = list(grouped.items())

    for i, segment in enumerate(data):
        midi = pretty_midi.PrettyMIDI(initial_tempo=tempo, resolution=args.resolution)

        if args.range and args.range[0]:
            i += args.range[0]
        index = str(i).zfill(fill_length)
        if isinstance(segment, list):
            notes = segment
            fname = f'{index}.mid'
        elif isinstance(segment, tuple) and len(segment) == 2:
            segment_id, notes = segment
            if not isinstance(segment_id, str):
                segment_id = '_'.join(str(x) for x in segment_id)
            fname = f'{index}_{segment_id}.mid'
        else:
            raise RuntimeError(f'Cannot parse segment: {segment}')

        if args.stretch is not None:
            for note in notes:
                note.start *= args.stretch
                note.end *= args.stretch

        # Remove notes with length below the MIDI resolution
        notes = [note for note in notes
                 if midi.time_to_tick(note.start) < midi.time_to_tick(note.end)]

        # Some notes might be overlapping, we need to split them between multiple tracks.
        # TODO: This calls for a more efficient implementation.
        tracks = [[]]
        for note in notes:
            for track in tracks:
                # Find the first track without an overlapping note.
                if not any(note2.pitch == note.pitch
                           and note2.start < note.end
                           and note2.end > note.start
                           for note2 in track):
                    track.append(note)
                    break

            # Always keep the last track empty (we delete it afterwards)
            if tracks[-1]:
                tracks.append([])
        del tracks[-1]

        midi.time_signature_changes[:] = [pretty_midi.TimeSignature(*args.time_signature, 0.)]

        for track in tracks:
            instrument = pretty_midi.Instrument(name=args.instrument,
                                                program=args.program,
                                                is_drum=args.drums)
            instrument.notes[:] = track
            midi.instruments.append(instrument)

        midi.write(os.path.join(args.output_dir, fname))


def _shift_and_clip(note, start, end):
    return pretty_midi.Note(
        start=start + max(note.start, 0),
        end=min(start + note.end, end),
        pitch=note.pitch,
        velocity=note.velocity)


if __name__ == '__main__':
    main()
