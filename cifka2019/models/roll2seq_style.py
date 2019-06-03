#!/usr/bin/env python3
import argparse
import collections
import glob
import json
import os
import pickle

import coloredlogs
import numpy as np
import scipy.spatial.distance as scipy_distance
import tensorflow as tf

from museflow import logger
from museflow.components import EmbeddingLayer, RNNLayer, RNNDecoder
from museflow.config import Configuration, configurable
from museflow.model_utils import (DatasetManager, create_train_op, prepare_train_and_val_data,
                                  make_simple_dataset, set_random_seed)
from museflow.nn.rnn import InputWrapper
from museflow.trainer import BasicTrainer
from museflow.vocabulary import Vocabulary

from cifka2019.eval.notes_chroma_similarity import chroma_similarity
from cifka2019.eval.style_profile import time_pitch_diff_hist


@configurable(['embedding_layer', 'style_embedding_layer', 'style_projection', '2d_layers',
               '1d_layers', 'encoder', 'state_projection', 'decoder', 'attention_mechanism',
               'training'])
class CNNRNNSeq2Seq:

    def __init__(self, dataset_manager, train_mode, vocabulary, style_vocabulary,
                 sampling_seed=None):
        self._cfg = _LegacyConfiguration(self._cfg)
        self._train_mode = train_mode
        self._is_training = tf.placeholder_with_default(False, [], name='is_training')

        self.dataset_manager = dataset_manager

        inputs, self.style_id, decoder_inputs, decoder_targets = self.dataset_manager.get_batch()
        batch_size = tf.shape(inputs)[0]

        layers_2d = self._cfg.maybe_configure('2d_layers') or []
        layers_1d = self._cfg.maybe_configure('1d_layers') or []

        features = inputs
        if layers_2d:
            # Expand to 4 dimensions: [batch_size, rows, time, channels]
            features = tf.expand_dims(features, -1)

            # 2D layers: 4 -> 4 dimensions
            for layer in layers_2d:
                logger.debug(f'Inputs to layer {layer} have shape {features.shape}')
                features = self._apply_layer(layer, features)
            logger.debug(f'After the 2D layers, the features have shape {features.shape}')

            # Features have shape [batch_size, rows, time, channels]. Switch rows and cols, then
            # flatten rows and channels to get 3 dimensions: [batch_size, time, new_channels].
            features = tf.transpose(features, perm=[0, 2, 1, *range(3, features.shape.ndims)])
            num_channels = features.shape[2] * features.shape[3]
            features = tf.reshape(features, [batch_size, -1, num_channels])

        # 1D layers: 3 -> 3 dimensions: [batch_size, time, channels]
        for layer in layers_1d:
            logger.debug(f'Inputs to layer {layer} have shape {features.shape}')
            features = self._apply_layer(layer, features)

        encoder = self._cfg.configure('encoder', RNNLayer,
                                      training=self._is_training,
                                      name='encoder')
        encoder_states, encoder_final_state = encoder.apply(features)

        embeddings = self._cfg.configure('embedding_layer', EmbeddingLayer,
                                         input_size=len(vocabulary))

        # If the style representation is sparse, we do embedding lookup, otherwise we apply
        # a projection (a dense layer).
        if self.style_id.dtype.is_integer:
            style_layer = self._cfg.configure('style_embedding_layer', EmbeddingLayer,
                                              input_size=len(style_vocabulary),
                                              name='style_embedding')
        else:
            style_layer = self._cfg.configure('style_projection', tf.layers.Dense,
                                              name='style_projection')
        self.style_vector = style_layer(self.style_id)

        def cell_wrap_fn(cell):
            """Wrap the RNN cell in order to pass the style embedding as input."""
            return InputWrapper(cell, input_fn=lambda _: self.style_vector)

        with tf.variable_scope('attention'):
            attention = self._cfg.maybe_configure('attention_mechanism', memory=encoder_states)
        self.decoder = self._cfg.configure('decoder', RNNDecoder,
                                           vocabulary=vocabulary,
                                           embedding_layer=embeddings,
                                           attention_mechanism=attention,
                                           cell_wrap_fn=cell_wrap_fn,
                                           training=self._is_training)

        state_projection = self._cfg.configure('state_projection', tf.layers.Dense,
                                               units=self.decoder.initial_state_size,
                                               name='state_projection')
        decoder_initial_state = state_projection(encoder_final_state)

        # Build the training version of the decoder and the training ops
        self.training_ops = None
        if train_mode:
            _, self.loss = self.decoder.decode_train(decoder_inputs, decoder_targets,
                                                     initial_state=decoder_initial_state)
            self.training_ops = self._make_train_ops()

        # Build the sampling and greedy version of the decoder
        self.softmax_temperature = tf.placeholder(tf.float32, [], name='softmax_temperature')
        self.sample_outputs, self.sample_final_state = self.decoder.decode(
            mode='sample',
            softmax_temperature=self.softmax_temperature,
            initial_state=decoder_initial_state,
            batch_size=batch_size,
            random_seed=sampling_seed)
        self.greedy_outputs, self.greedy_final_state = self.decoder.decode(
            mode='greedy',
            initial_state=decoder_initial_state,
            batch_size=batch_size)

    def _make_train_ops(self):
        train_op = self._cfg.configure('training', create_train_op, loss=self.loss)
        init_op = tf.global_variables_initializer()

        tf.summary.scalar('train/loss', self.loss)
        train_summary_op = tf.summary.merge_all()

        return BasicTrainer.TrainingOps(loss=self.loss,
                                        train_op=train_op,
                                        init_op=init_op,
                                        summary_op=train_summary_op,
                                        training_placeholder=self._is_training)

    def _apply_layer(self, layer, features):
        if isinstance(layer, (tf.layers.Dropout, tf.keras.layers.Dropout)):
            return layer(features, training=self._is_training)
        return layer(features)

    def run(self, session, dataset, sample=False, softmax_temperature=1.):
        _, output_ids_tensor = self.sample_outputs if sample else self.greedy_outputs

        return self.dataset_manager.run_over_dataset(
            session, output_ids_tensor, dataset,
            feed_dict={self.softmax_temperature: softmax_temperature},
            concat_batches=True)


@configurable(pass_kwargs=False)
class TranslationExperiment:

    def __init__(self, logdir, train_mode):
        set_random_seed(self._cfg.get('random_seed', None))

        self.input_encoding = self._cfg['input_encoding'].configure()
        self.output_encoding = self._cfg['output_encoding'].configure()
        with open(self._cfg.get('style_list')) as f:
            style_list = [line.rstrip('\n') for line in f]
        self.style_vocabulary = Vocabulary(style_list, pad_token=None, start_token=None, end_token=None)

        input_dtype = self._cfg.get('input_dtype', tf.float32)
        num_rows = getattr(self.input_encoding, 'num_rows', None)
        self.input_shapes = (([num_rows, None] if num_rows else [None]), [], [None], [None])
        self.input_types = (input_dtype, tf.int32, tf.int32, tf.int32)
        self.dataset_manager = DatasetManager(
            output_types=self.input_types,
            output_shapes=tuple([None, *shape] for shape in self.input_shapes))

        self.model = self._cfg['model'].configure(CNNRNNSeq2Seq,
                                                  dataset_manager=self.dataset_manager,
                                                  train_mode=train_mode,
                                                  vocabulary=self.output_encoding.vocabulary,
                                                  style_vocabulary=self.style_vocabulary)
        self.trainer = self._cfg['trainer'].configure(BasicTrainer,
                                                      dataset_manager=self.dataset_manager,
                                                      training_ops=self.model.training_ops,
                                                      logdir=logdir,
                                                      write_summaries=train_mode)

        if train_mode:
            # Configure the dataset manager with the training and validation data.
            self._cfg['data_prep'].configure(
                prepare_train_and_val_data,
                dataset_manager=self.dataset_manager,
                train_generator=self._cfg['train_data'].configure(self._load_data,
                                                                  log=True, limit_length=True),
                val_generator=self._cfg['val_data'].configure(self._load_data),
                output_types=self.input_types,
                output_shapes=self.input_shapes)

    def train(self, args):
        val_inputs, val_target_styles, val_targets = zip(
            *self._cfg['val_data'].configure(self._load_data, encode=False)())

        style_profiles = {}
        if 'style_profiles_dir' in self._cfg:
            for style in set(val_target_styles):
                profiles_path = os.path.join(self._cfg.get('style_profiles_dir'), f'{style}.json')
                with open(profiles_path) as f:
                    style_profiles[style] = {k: np.array(v) for k, v in json.load(f).items()}

        logger.info("Starting training.")
        for state in self.trainer.iter_train(period=self._cfg.get('evaluation_period', 0)):
            if state.step == 0:
                continue

            for mode in ['greedy', 'sample']:
                # Run model on validation data
                output_ids = self.model.run(self.trainer.session, 'val', sample=(mode == 'sample'))
                outputs = [self.output_encoding.decode(seq) for seq in output_ids]
                assert len(outputs) == len(val_inputs)

                # Compute cosine similarity of chroma features
                similarity = np.mean([
                    chroma_similarity(a, b, sampling_rate=12, window_size=24, stride=12)
                    for a, b in zip(val_inputs, outputs)])
                self.trainer.write_scalar_summary(f'val/{mode}/chroma_similarity', similarity)

                # Compute cosine similarity of style profiles
                if style_profiles:
                    outputs_by_style = collections.defaultdict(list)
                    for output_notes, style in zip(outputs, val_target_styles):
                        outputs_by_style[style].append(output_notes)
                    for max_time in [2, 4]:
                        distances = []
                        for style in outputs_by_style:
                            out_profile = time_pitch_diff_hist(
                                outputs_by_style[style],
                                max_time=max_time, bin_size=1/6, pitch_range=20)
                            ref_profile = (
                                style_profiles[style][f'time_pitch_diff_hist_t{max_time}_f6'])
                            distance = scipy_distance.cosine(out_profile.reshape(-1),
                                                             ref_profile.reshape(-1))
                            distances.append(distance if not np.isnan(distance) else 0.)
                        self.trainer.write_scalar_summary(
                            f'val/{mode}/style_similarity_t{max_time}', 1. - np.mean(distances))

    def run(self, args):
        self.trainer.load_variables(checkpoint_file=args.checkpoint)
        data = pickle.load(args.input_file)

        def generator():
            style_id = self.style_vocabulary.to_id(args.target_style)
            for example in data:
                segment_id, notes = example
                yield self.input_encoding.encode(notes), style_id, [], []

        dataset = make_simple_dataset(
            generator,
            output_types=self.input_types,
            output_shapes=self.input_shapes,
            batch_size=args.batch_size)

        output_ids = self.model.run(self.trainer.session, dataset, args.sample, args.softmax_temperature)
        outputs = [(segment_id, self.output_encoding.decode(seq))
                   for seq, (segment_id, _) in zip(output_ids, data)]

        pickle.dump(outputs, args.output_file)

    def _load_data(self, paths=None, src=None, tgt=None, log=False, encode=True, limit_length=False):
        if paths is not None:
            src, tgt = paths, paths
        del paths
        if isinstance(src, str):
            src = [src]
        if isinstance(tgt, str):
            tgt = [tgt]

        def glob_paths(patterns):
            paths = [path
                     for pattern in patterns
                     for path in sorted(glob.glob(pattern, recursive=True))]
            if not paths:
                raise RuntimeError(f'Pattern list {patterns} did not match any paths.')
            return paths

        def make_data_dict(data):
            # data is a list of tuples (segment_id, notes).
            # segment_id is a tuple (song_and_style, start, end).
            # song_and_style consists of the song name and the style, separated by a dot.
            keys = []
            data_defaultdict = collections.defaultdict(list)
            for (song_and_style, start, end), val in data:
                song_name, style = song_and_style.rsplit('.', maxsplit=1)
                new_key = (song_name, start, end)
                data_defaultdict[new_key].append((style, val))
                keys.append(new_key)
            return dict(data_defaultdict), keys

        def get_pairs(src_data, tgt_data):
            src_data, keys = make_data_dict(src_data)
            tgt_data, _ = make_data_dict(tgt_data)

            for key in keys:
                for src_style, src_notes in src_data[key]:
                    if limit_length and len(src_notes) > self._cfg.get('max_src_notes', np.inf):
                        logger.warning(f'Skipping source segment {key}, {src_style} with '
                                       f'{len(src_notes)} source notes')
                        continue

                    try:
                        tgt_segments = tgt_data[key]
                    except KeyError as e:
                        logger.warning(f'KeyError: {e}')
                        continue

                    for tgt_style, tgt_notes in tgt_segments:
                        if tgt_style == src_style:
                            continue
                        if limit_length and len(tgt_notes) > self._cfg.get('max_tgt_notes', np.inf):
                            logger.warning(f'Skipping target segment {key}, {tgt_style} with '
                                           f'{len(tgt_notes)} notes')
                            continue

                        if encode:
                            src_ids = self.input_encoding.encode(src_notes)
                            tgt_ids = self.output_encoding.encode(tgt_notes, add_start=True, add_end=True)
                            tgt_style_id = self.style_vocabulary.to_id(tgt_style)
                            yield src_ids, tgt_style_id, tgt_ids[:-1], tgt_ids[1:]
                        else:
                            yield src_notes, tgt_style, tgt_notes

        def generator():
            i = 0
            for src_path, tgt_path in zip(glob_paths(src), glob_paths(tgt)):
                if log:
                    logger.debug(f'Reading from {src_path}, {tgt_path}')
                with open(src_path, 'rb') as f:
                    src_data = pickle.load(f)
                with open(tgt_path, 'rb') as f:
                    tgt_data = pickle.load(f)
                for item in get_pairs(src_data, tgt_data):
                    i += 1
                    yield item
            if log:
                logger.info('Done loading data ({} examples)'.format(i))

        return generator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, required=True, help='model directory')
    parser.set_defaults(train_mode=False)
    subparsers = parser.add_subparsers(title='action')

    subparser = subparsers.add_parser('train')
    subparser.set_defaults(func=TranslationExperiment.train, train_mode=True)

    subparser = subparsers.add_parser('run')
    subparser.set_defaults(func=TranslationExperiment.run)
    subparser.add_argument('input_file', type=argparse.FileType('rb'), metavar='INPUTFILE')
    subparser.add_argument('output_file', type=argparse.FileType('wb'), metavar='OUTPUTFILE')
    subparser.add_argument('target_style', type=str, metavar='STYLE')
    subparser.add_argument('--checkpoint', default=None, type=str)
    subparser.add_argument('--batch-size', default=32, type=int)
    subparser.add_argument('--sample', action='store_true')
    subparser.add_argument('--softmax-temperature', default=1., type=float)
    args = parser.parse_args()

    config_file = os.path.join(args.logdir, 'model.yaml')
    with open(config_file, 'rb') as f:
        config = Configuration.from_yaml(f)
    logger.debug(config)

    experiment = config.configure(TranslationExperiment,
                                  logdir=args.logdir, train_mode=args.train_mode)
    args.func(experiment, args)


class _LegacyConfiguration:

    def __init__(self, cfg):
        self.cfg = cfg

    def configure(self, *args, **kwargs):
        key, *args = args
        return self.cfg[key].configure(*args, **kwargs)

    def maybe_configure(self, *args, **kwargs):
        key, *args = args
        return self.cfg[key].maybe_configure(*args, **kwargs)


if __name__ == '__main__':
    coloredlogs.install(level='DEBUG', logger=logger, isatty=True)
    main()
