#!/usr/bin/env python3
import argparse
import re
import sys

import mido


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('input_file', type=str, metavar='INPUTFILE')
    parser.add_argument('output_file', type=str, metavar='OUTPUTFILE')
    parser.add_argument('-p', '--program',  type=lambda l: [int(x) for x in l.split(',')],
                        dest='programs', default=None)
    parser.add_argument('-t', '--track-re', type=str, default=None)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-c', '--channel', type=lambda l: [int(x) for x in l.split(',')],
                       dest='channels', default=None)
    group.add_argument('-C', '--except-channel', type=lambda l: [int(x) for x in l.split(',')],
                       dest='except_channels', default=None)
    args = parser.parse_args()

    if args.channels is not None:
        channels = [i - 1 for i in args.channels]
    elif args.except_channels is not None:
        channels = [i for i in range(16) if i + 1 not in args.except_channels]
    else:
        channels = range(16)

    if args.programs is not None:
        programs = [i - 1 for i in args.programs]
    else:
        programs = range(128)

    midi = mido.MidiFile(args.input_file)
    has_notes = False
    
    for track in midi.tracks:
        messages = []
        delta_time = 0

        track_match = args.track_re is None or re.search(args.track_re, track.name)
        channel_program = {i: 0 for i in range(16)}

        for msg in track:
            if msg.type == 'program_change':
                channel_program[msg.channel] = msg.program

            if (not hasattr(msg, 'channel')
                or (track_match
                    and channel_program[msg.channel] in programs
                    and msg.channel in channels)):
                msg.time += delta_time
                messages.append(msg)
                delta_time = 0

                has_notes = has_notes or msg.type in ['note_on', 'note_off']
            else:
                delta_time += msg.time

        track[:] = messages

    if has_notes:
        midi.save(args.output_file)
    else:
        print(f'Ignoring file {args.input_file} with no music to output', file=sys.stderr)


if __name__ == '__main__':
    main()
