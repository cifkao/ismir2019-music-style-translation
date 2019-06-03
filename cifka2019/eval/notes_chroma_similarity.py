#!/usr/bin/env python3
"""Measure chroma vector similarity ("content preservation").

Described in "Transferring The Style of Homophonic Music Using Recurrent Neural Networks and
Autoregressive Models", Wei-Tsung Lu and Li Su, ISMIR 2018.
http://ismir2018.ircam.fr/doc/pdfs/107_Paper.pdf
"""

import argparse
import pickle

import numpy as np
import pretty_midi
import scipy.signal
import scipy.spatial.distance


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('file1', type=argparse.FileType('rb'))
    parser.add_argument('file2', type=argparse.FileType('rb'))
    parser.add_argument('-f', '--sampling-rate', type=int, required=True)
    parser.add_argument('-w', '--window', type=int, required=True)
    parser.add_argument('-s', '--stride', type=int, default=1)
    parser.add_argument('--average', action='store_true')
    parser.add_argument('--no-velocity', action='store_true')
    args = parser.parse_args()

    data_a, data_b = (pickle.load(f) for f in (args.file1, args.file2))

    similarities = []
    for (id_a, notes_a), (id_b, notes_b) in zip(data_a, data_b):
        if id_a != id_b:
            raise RuntimeError(f'{id_a} != {id_b}')

        similarity = chroma_similarity(notes_a, notes_b,
                                       sampling_rate=args.sampling_rate,
                                       window_size=args.window,
                                       stride=args.stride,
                                       use_velocity=not args.no_velocity)

        similarities.append(similarity)
        if not args.average:
            print(id_a, similarity, sep='\t')

    if args.average:
        print(np.mean(similarities))


def chroma_similarity(notes_a, notes_b, sampling_rate, window_size, stride, use_velocity=False):
    if not use_velocity:
        notes_a, notes_b = (_strip_velocity(notes) for notes in (notes_a, notes_b))

    chroma_a, chroma_b = (_get_chroma(notes, sampling_rate) for notes in (notes_a, notes_b))

    # Make sure the chroma matrices have the same dimensions.
    if chroma_a.shape[1] < chroma_b.shape[1]:
        chroma_a, chroma_b = chroma_b, chroma_a
    chroma_b = np.pad(chroma_b, [(0, 0), (0, chroma_a.shape[1] - chroma_b.shape[1])],
                      mode='constant')

    # Compute a moving average over time.
    avg_filter = np.ones((1, window_size)) / window_size
    chroma_avg_a, chroma_avg_b = (_convolve_strided(chroma, avg_filter, stride)
                                  for chroma in (chroma_a, chroma_b))

    return _average_cos_similarity(chroma_avg_a, chroma_avg_b)


def _strip_velocity(notes):
    return [pretty_midi.Note(pitch=n.pitch, start=n.start, end=n.end, velocity=127)
            for n in notes]


def _get_chroma(notes, sampling_rate):
    midi = pretty_midi.Instrument(0)
    midi.notes[:] = notes
    return midi.get_chroma(fs=sampling_rate)


def _average_cos_similarity(chroma_a, chroma_b):
    """Compute the column-wise cosine similarity, averaged over all non-zero columns."""
    nonzero_cols_ab = []
    for chroma in (chroma_a, chroma_b):
        col_norms = np.linalg.norm(chroma, axis=0)
        nonzero_cols = col_norms > 1e-9
        nonzero_cols_ab.append(nonzero_cols)
        # Note: 'np.divide' needs the 'out' parameter, otherwise the output would get written to
        # an uninitialized array.
        np.divide(chroma, col_norms, where=nonzero_cols, out=chroma)

    # Count the columns where at least one of the two matrices is nonzero.
    num_nonzero_cols = np.logical_or(*nonzero_cols_ab).sum()

    # Compute the dot product.
    return np.tensordot(chroma_a, chroma_b) / num_nonzero_cols


def _convolve_strided(data, filtr, stride):
    """Compute a 2D convolution with the given stride along the second dimension.

    A full (zero-padded) 2D convolution is computed, then subsampled according to the stride with
    an offset calculated so that the convolution window is aligned to the left edge of the original
    array.
    """
    convolution = scipy.signal.convolve2d(data, filtr, mode='full')
    offset = (filtr.shape[-1] - 1) % stride  # Make sure the windows are aligned
    return convolution[:, offset::stride]


if __name__ == '__main__':
    main()
