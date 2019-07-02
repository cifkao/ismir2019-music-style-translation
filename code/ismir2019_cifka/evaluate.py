#!/usr/bin/env python3
import argparse
import collections
import itertools
import json
import logging
import os
import pickle
import warnings

import coloredlogs
import numpy as np
import pandas as pd
import sklearn.metrics.pairwise

from ismir2019_cifka.eval.style_profile import time_pitch_diff_hist
from ismir2019_cifka.eval.notes_chroma_similarity import chroma_similarity


CHROMA_SIMILARITY_PARAMS = dict(sampling_rate=12, window_size=24, stride=12, use_velocity=False)
CHROMA_SIMILARITY_PARAMS_2 = dict(sampling_rate=12, window_size=48, stride=48, use_velocity=False)
STYLE_SIMILARITY_PARAMS = dict(max_time=4, bins_per_beat=6)


def main():
    coloredlogs.install(level='DEBUG', logger=logging.getLogger(), isatty=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--models-dir', type=str, metavar='DIR', default='.',
                        help='the directory containing all the model directories')
    parser.add_argument('--style-list', type=str, metavar='FILE', required=True,
                        help='a file with the list of all the styles')
    parser.add_argument('--style-profile-dir', type=str, metavar='DIR', required=True,
                        help='the style profile directory; expected to contain a file named '
                             'STYLE.json for each style')
    parser.add_argument('--models', type=str, nargs='+', metavar='DIR', required=True,
                        help='the model names (directories in models-dir)')
    parser.add_argument('--data-prefix', type=str, required=True, metavar='PREFIX',
                        help='the prefix for the pickle filename containing the model outputs, '
                             'relative to the model directory; will be suffixed with STYLE.pickle')
    parser.add_argument('--instrument', type=str,
                        help='the instrument, e.g. Bass, Piano')
    parser.add_argument('--source', type=str, required=True, metavar='FILE',
                        help='the file containing the source track segments (for computing the '
                             'chroma similarities)')
    parser.add_argument('--reference-dir', type=str, metavar='DIR',
                        help='the root of reference files; expected to contain files named '
                             'STYLE/INSTRUMENT.pickle')
    parser.add_argument('--styles', type=str, nargs='+', metavar='STYLE',
                        help='names of styles to evaluate on')
    parser.add_argument('--output', type=str, required=True, metavar='FILE',
                        help='a pickle file to contain the results')
    args = parser.parse_args()

    if args.styles:
        styles = args.styles
    else:
        with open(args.style_list) as f:
            styles = [line.rstrip('\n') for line in f]

    style_profiles = {}
    for style in styles:
        with open(os.path.join(args.style_profile_dir, style + '.json'), 'rb') as f:
            style_profiles[style] = {k: np.array(v).reshape(-1) for k, v in json.load(f).items()}

    with open(args.source, 'rb') as f:
        sources = pickle.load(f)

    style_eval = StyleProfileEvaluator(style_profiles, **STYLE_SIMILARITY_PARAMS)
    chroma_eval = ChromaEvaluator(sources, **CHROMA_SIMILARITY_PARAMS)
    chroma_eval2 = ChromaEvaluator(sources, **CHROMA_SIMILARITY_PARAMS_2)

    for style in styles:
        style_eval.evaluate('source', sources, style)
    chroma_eval.evaluate('source', sources, None)
    chroma_eval2.evaluate('source', sources, None)

    if args.reference_dir:
        if not args.instrument:
            raise ValueError('Instrument not specified.')

        sources_no_style = [((name.rsplit('.', maxsplit=1)[0], start, end), notes)
                            for (name, start, end), notes in sources]
        keys, values = (list(x) for x in zip(*sources_no_style))
        np.random.seed(1234)
        np.random.shuffle(values)
        sources_no_style_shuf = list(zip(keys, values))

        for style in styles:
            with open(os.path.join(args.reference_dir, style, f'{args.instrument}.pickle'), 'rb') as f:
                references = pickle.load(f)

            style_eval.evaluate('reference', references, style)
            chroma_eval.evaluate('reference', references, style, references=sources_no_style)
            chroma_eval2.evaluate('reference', references, style, references=sources_no_style)

            chroma_eval.evaluate('random', references, style, references=sources_no_style_shuf)
            chroma_eval2.evaluate('random', references, style, references=sources_no_style_shuf)

    for model_name in args.models:
        model_path = os.path.join(args.models_dir, model_name)

        for style in styles:
            with open(os.path.join(model_path, args.data_prefix + style + '.pickle'), 'rb') as f:
                data = pickle.load(f)

            style_eval.evaluate(model_name, data, style)
            chroma_eval.evaluate(model_name, data, style)
            chroma_eval2.evaluate(model_name, data, style)

    style_results, song_style_results = style_eval.get_results(average=False)
    chroma_results = chroma_eval.get_results(average=False)
    chroma2_results = chroma_eval2.get_results(average=False)

    with open(args.output, 'wb') as f:
        pickle.dump((style_results, song_style_results, chroma_results, chroma2_results), f)


class ChromaEvaluator:

    def __init__(self, references, **kwargs):
        self._references = _group_by_segment_id(references)
        self._kwargs = kwargs
        self._similarities = []

        if any(len(x) > 1 for x in self._references.values()):
            warnings.warn('Repeated keys in reference', RuntimeWarning)

    def evaluate(self, name, data, style, references=None):
        if len(dict(data)) != len(data):
            raise RuntimeError('Repeated keys in data')

        references = _group_by_segment_id(references) if references else self._references

        missing_keys = []
        for segment_id, notes in data:
            try:
                ref_segments = references[segment_id]
            except KeyError:
                missing_keys.append(segment_id)
                continue

            for ref_notes in ref_segments:
                similarity = chroma_similarity(ref_notes, notes, **self._kwargs)
                self._similarities.append({'name': name, 'chroma_sim': similarity,
                                           'song_name': segment_id[0], 'style': style})

        if missing_keys:
            warnings.warn(f'{name}: {len(missing_keys)} missing keys', RuntimeWarning)

    def get_results(self, average=True):
        similarities = pd.DataFrame(self._similarities)
        if not average:
            return similarities
        return similarities.groupby('name').mean()


class StyleProfileEvaluator:

    def __init__(self, style_profiles, max_time, bins_per_beat):
        self._profiles = style_profiles
        self._max_time = max_time
        self._bins_per_beat = bins_per_beat

        self._similarities = []
        self._per_song_similarities = []

    def evaluate(self, name, data, style):
        ref_profile = self.get_profile(style)

        # Compute overall similarity
        profile = self.compute_profile([notes for _, notes in data])
        [[similarity]] = sklearn.metrics.pairwise.cosine_similarity([profile], [ref_profile])
        self._similarities.append({'name': name, 'style_sim': similarity, 'style': style})

        # Compute similarity for each song
        for song, segments in itertools.groupby(data, lambda x: x[0][0]):
            profile = self.compute_profile([notes for _, notes in segments])
            [[similarity]] = sklearn.metrics.pairwise.cosine_similarity([profile], [ref_profile])
            self._per_song_similarities.append(
                {'name': name, 'song_style_sim': similarity, 'style': style, 'song_name': song})

    def get_results(self, average=True):
        similarities = pd.DataFrame(self._similarities)
        per_song_similarities = pd.DataFrame(self._per_song_similarities)

        if not average:
            return similarities, per_song_similarities

        return pd.concat([
            similarities.groupby('name').mean(),
            per_song_similarities.groupby('name').mean(),
            per_song_similarities.groupby('name').std().rename(
                columns={'song_style_sim': 'song_style_sim_err'})
        ], axis=1)

    def get_profile(self, style):
        return self._profiles[style][f'time_pitch_diff_hist_t{self._max_time}_f{self._bins_per_beat}']

    def compute_profile(self, data):
        return np.nan_to_num(
            time_pitch_diff_hist(data, max_time=self._max_time, bin_size=1/self._bins_per_beat,
                                 normed=True).reshape(-1))


def _group_by_segment_id(data):
    return {
        segment_id: [notes for _, notes in segments]
        for segment_id, segments in itertools.groupby(data, lambda x: x[0])
    }


if __name__ == '__main__':
    main()
