#!/usr/bin/env python3
import argparse
import json
import sys

import numpy as np
import pretty_midi

from museflow.scripts.chop_midi import normalize_tempo


def time_pitch_diff_hist(data, max_time=2, bin_size=1/6, pitch_range=20, normed=False):
    """Compute an onset-time-difference vs. interval histogram.

    Args:
        data: A list of lists of `pretty_midi.Note`.
        max_time: The maximum time between two notes to be considered.
        bin_size: The bin size along the time axis.
        pitch_range: The number of pitch difference bins in each direction (positive or negative,
            excluding 0). Each bin has size 1.
        normed: Whether to normalize the histogram.

    Returns:
        A 2D `np.array` of shape `[max_time / bin_size, 2 * pitch_range + 1]`.
    """
    epsilon = 1e-9
    time_diffs = []
    intervals = []
    for notes in data:
        onsets = [n.start for n in notes]
        diff_mat = np.subtract.outer(onsets, onsets)

        # Count only positive time differences.
        index_pairs = zip(*np.where((diff_mat < max_time - epsilon) & (diff_mat >= 0.)))
        for j, i in index_pairs:
            if j == i:
                continue

            time_diffs.append(diff_mat[j, i])
            intervals.append(notes[j].pitch - notes[i].pitch)

    histogram, _, _ = np.histogram2d(intervals, time_diffs, normed=normed,
                                     bins=[np.arange(-(pitch_range + 1), pitch_range + 1) + 0.5,
                                           np.arange(0., max_time + bin_size - epsilon, bin_size)])
    return histogram


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('midi_paths', nargs='+', metavar='MIDIFILE')
    args = parser.parse_args()

    results = {}

    def get_data():
        for path in args.midi_paths:
            midi = pretty_midi.PrettyMIDI(path)
            normalize_tempo(midi, 60)
            yield [note for instr in midi.instruments for note in instr.notes]

    results['time_pitch_diff_hist_t2_f6'] = time_pitch_diff_hist(
        get_data(), max_time=2, bin_size=1/6, normed=True)
    results['time_pitch_diff_hist_t4_f6'] = time_pitch_diff_hist(
        get_data(), max_time=4, bin_size=1/6, normed=True)

    json.dump(results, sys.stdout, default=lambda a: a.tolist())
    sys.stdout.write('\n')


if __name__ == '__main__':
    main()
