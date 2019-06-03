#!/usr/bin/env python3
import argparse
import os
import pickle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    parser.add_argument('--strip-dirs', action='store_true')
    parser.add_argument('--strip-exts', type=int, default=0)
    args = parser.parse_args()

    with open(args.filename, 'rb') as f:
        data = pickle.load(f)

    def adjust_path(path):
        if args.strip_dirs:
            path = os.path.basename(path)
        for _ in range(args.strip_exts):
            path, _ = os.path.splitext(path)
        return path

    data = [((adjust_path(path), *rest), item)
            for (path, *rest), item in data]

    with open(args.filename, 'wb') as f:
        pickle.dump(data, f)


if __name__ == '__main__':
    main()
