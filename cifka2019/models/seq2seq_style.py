#!/usr/bin/env python3
import argparse
import collections
import glob
import os
import pickle
import shutil

import coloredlogs
import numpy as np
import tensorflow as tf

from museflow import logger
from museflow.components import EmbeddingLayer, RNNLayer, RNNDecoder
from museflow.config import Configuration, configurable
from museflow.model_utils import (DatasetManager, create_train_op, prepare_train_and_val_data,
                                  make_simple_dataset, set_random_seed)
from museflow.nn.rnn import InputWrapper
from museflow.trainer import BasicTrainer
from museflow.vocabulary import Vocabulary


@configurable(['embedding_layer', 'style_embedding_layer', 'encoder', 'state_projection',
               'decoder', 'attention_mechanism', 'training'])
class RNNSeq2Seq:

    def __init__(self, train_mode, vocabulary, style_vocabulary, sampling_seed=None):
        self._cfg = _LegacyConfiguration(self._cfg)
        self._train_mode = train_mode
        self._is_training = tf.placeholder_with_default(False, [], name='is_training')

        self.dataset_manager = DatasetManager(
            output_types=(tf.int32, tf.int32, tf.int32, tf.int32),
            output_shapes=([None, None], [None], [None, None], [None, None]))

        inputs, self.style_id, decoder_inputs, decoder_targets = self.dataset_manager.get_batch()
        batch_size = tf.shape(inputs)[0]

        embeddings = self._cfg.configure('embedding_layer', EmbeddingLayer,
                                         input_size=len(vocabulary))
        encoder = self._cfg.configure('encoder', RNNLayer,
                                      training=self._is_training,
                                      name='encoder')
        encoder_states, encoder_final_state = encoder.apply(embeddings.embed(inputs))

        style_embeddings = self._cfg.configure('style_embedding_layer', EmbeddingLayer,
                                               input_size=len(style_vocabulary),
                                               name='style_embedding')
        self.style_vector = style_embeddings.embed(self.style_id)
        def cell_wrap_fn(cell):
            """Wrap the RNN cell in order to pass the style embedding as input."""
            return InputWrapper(cell, input_fn=lambda _: self.style_vector)

        with tf.variable_scope('attention'):
            attention = self._cfg.maybe_configure('attention_mechanism', memory=encoder_states)
        decoder = self._cfg.configure('decoder', RNNDecoder,
                                      vocabulary=vocabulary,
                                      embedding_layer=embeddings,
                                      attention_mechanism=attention,
                                      cell_wrap_fn=cell_wrap_fn,
                                      training=self._is_training)

        state_projection = self._cfg.configure('state_projection', tf.layers.Dense,
                                               units=decoder.initial_state_size,
                                               name='state_projection')
        decoder_initial_state = state_projection(encoder_final_state)

        # Build the training version of the decoder and the training ops
        self.training_ops = None
        if train_mode:
            _, self.loss = decoder.decode_train(decoder_inputs, decoder_targets,
                                                initial_state=decoder_initial_state)
            self.training_ops = self._make_train_ops()

        # Build the sampling and greedy version of the decoder
        self.softmax_temperature = tf.placeholder(tf.float32, [], name='softmax_temperature')
        self.sample_outputs, _ = decoder.decode(mode='sample',
                                                softmax_temperature=self.softmax_temperature,
                                                initial_state=decoder_initial_state,
                                                batch_size=batch_size,
                                                random_seed=sampling_seed)
        self.greedy_outputs, _ = decoder.decode(mode='greedy',
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

    def run(self, session, dataset, sample=False, softmax_temperature=1.):
        _, output_ids_tensor = self.sample_outputs if sample else self.greedy_outputs

        return self.dataset_manager.run_over_dataset(
            session, output_ids_tensor, dataset,
            feed_dict={self.softmax_temperature: softmax_temperature},
            concat_batches=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='path to the YAML configuration file')
    parser.add_argument('--logdir', type=str, required=True, help='model directory')
    parser.set_defaults(train_mode=False)
    subparsers = parser.add_subparsers(title='action')

    subparser = subparsers.add_parser('train')
    subparser.set_defaults(func=_train, train_mode=True)

    subparser = subparsers.add_parser('run')
    subparser.set_defaults(func=_run)
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

    model, trainer, encoding, style_vocabulary = config.configure(
        _init, logdir=args.logdir, train_mode=args.train_mode)
    args.func(model, trainer, encoding, style_vocabulary, config, args)


@configurable(pass_kwargs=False)
def _init(cfg, logdir, train_mode, **kwargs):
    set_random_seed(cfg.get('random_seed', None))

    encoding = cfg['encoding'].configure()
    with open(cfg.get('style_list')) as f:
        style_list = [line.rstrip('\n') for line in f]
    style_vocabulary = Vocabulary(style_list, pad_token=None, start_token=None, end_token=None)

    model = cfg['model'].configure(RNNSeq2Seq, train_mode=train_mode,
                                   vocabulary=encoding.vocabulary,
                                   style_vocabulary=style_vocabulary)
    trainer = cfg['trainer'].configure(BasicTrainer,
                                       dataset_manager=model.dataset_manager,
                                       training_ops=model.training_ops,
                                       logdir=logdir,
                                       write_summaries=train_mode)

    if train_mode:
        # Configure the dataset manager with the training and validation data.
        cfg['data_prep'].configure(
            prepare_train_and_val_data,
            dataset_manager=model.dataset_manager,
            train_generator=cfg['train_data'].configure(
                _load_data,
                encoding=encoding, style_vocabulary=style_vocabulary, config=cfg,
                log=True, limit_length=True),
            val_generator=cfg['val_data'].configure(
                _load_data,
                encoding=encoding, style_vocabulary=style_vocabulary, config=cfg),
            output_types=(tf.int32, tf.int32, tf.int32, tf.int32),
            output_shapes=([None], [], [None], [None]))

    return model, trainer, encoding, style_vocabulary


def _train(model, trainer, encoding, style_vocabulary, config, args):
    shutil.copy2(__file__, args.logdir)
    trainer.train()


def _run(model, trainer, encoding, style_vocabulary, config, args):
    trainer.load_variables(checkpoint_file=args.checkpoint)
    data = pickle.load(args.input_file)

    def generator():
        style_id = style_vocabulary.to_id(args.target_style)
        for example in data:
            segment_id, notes = example
            yield encoding.encode(notes, add_start=False, add_end=False), style_id, [], []

    dataset = make_simple_dataset(
        generator,
        output_types=(tf.int32, tf.int32, tf.int32, tf.int32),
        output_shapes=([None], [], [None], [None]),
        batch_size=args.batch_size)

    output_ids = model.run(trainer.session, dataset, args.sample, args.softmax_temperature)
    output = [(segment_id, encoding.decode(seq))
              for seq, (segment_id, _) in zip(output_ids, data)]

    pickle.dump(output, args.output_file)


def _load_data(paths, encoding, style_vocabulary, config, log=False, limit_length=False):
    if isinstance(paths, str):
        paths = [paths]

    def glob_paths(patterns):
        paths = [path
                 for pattern in patterns
                 for path in sorted(glob.glob(pattern, recursive=True))]
        if not paths:
            raise RuntimeError(f'Pattern list {patterns} did not match any paths.')
        return paths

    def get_pairs(data):
        # data is a list of tuples (segment_id, notes).
        # segment_id is a tuple (song_and_style, start, end).
        # song_and_style consists of the song name and the style, separated by a dot.
        data_defaultdict = collections.defaultdict(list)
        keys = []
        for (song_and_style, start, end), val in data:
            song_name, style = song_and_style.rsplit('.', maxsplit=1)
            new_key = (song_name, start, end)
            keys.append(new_key)
            data_defaultdict[new_key].append((style, val))
        data = dict(data_defaultdict)

        for key in keys:
            for src_style, src_notes in data[key]:
                if limit_length and len(src_notes) > config.get('max_src_notes', np.inf):
                    logger.warning(f'Skipping source segment {key}, {src_style} with '
                                   f'{len(src_notes)} source notes')
                    continue

                for tgt_style, tgt_notes in data[key]:
                    if tgt_style == src_style:
                        continue
                    if limit_length and len(tgt_notes) > config.get('max_tgt_notes', np.inf):
                        logger.warning(f'Skipping target segment {key}, {tgt_style} with '
                                       f'{len(tgt_notes)} notes')
                        continue

                    src_ids = encoding.encode(src_notes, add_start=False, add_end=False)
                    tgt_ids = encoding.encode(tgt_notes, add_start=True, add_end=True)
                    tgt_style_id = style_vocabulary.to_id(tgt_style)
                    yield src_ids, tgt_style_id, tgt_ids[:-1], tgt_ids[1:]

    def generator():
        i = 0
        for path in glob_paths(paths):
            if log:
                logger.debug(f'Reading from {path}')
            with open(path, 'rb') as f:
                data = pickle.load(f)
            for item in get_pairs(data):
                i += 1
                yield item
        if log:
            logger.info('Done loading data ({} examples)'.format(i))

    return generator


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
