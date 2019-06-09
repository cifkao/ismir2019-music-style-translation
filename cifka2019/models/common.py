import collections
import glob
import logging
import pickle


LOGGER = logging.getLogger('ismir2019_cifka')


def load_data(input_encoding, output_encoding, style_vocabulary,
              paths=None, src=None, tgt=None, log=False, encode=True):
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
                try:
                    tgt_segments = tgt_data[key]
                except KeyError as e:
                    LOGGER.warning(f'KeyError: {e}')
                    continue

                for tgt_style, tgt_notes in tgt_segments:
                    if tgt_style == src_style:
                        continue

                    if encode:
                        src_ids = input_encoding.encode(src_notes)
                        tgt_ids = output_encoding.encode(tgt_notes, add_start=True, add_end=True)
                        tgt_style_id = style_vocabulary.to_id(tgt_style)
                        yield src_ids, tgt_style_id, tgt_ids[:-1], tgt_ids[1:]
                    else:
                        yield src_notes, tgt_style, tgt_notes

    def generator():
        i = 0
        for src_path, tgt_path in zip(glob_paths(src), glob_paths(tgt)):
            if log:
                LOGGER.debug(f'Reading from {src_path}, {tgt_path}')
            with open(src_path, 'rb') as f:
                src_data = pickle.load(f)
            with open(tgt_path, 'rb') as f:
                tgt_data = pickle.load(f)
            for item in get_pairs(src_data, tgt_data):
                i += 1
                yield item
        if log:
            LOGGER.info('Done loading data ({} examples)'.format(i))

    return generator
