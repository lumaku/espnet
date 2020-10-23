#!/usr/bin/env python3

import argparse
import yaml

import kaldiio
import numpy as np

from kaldiio.utils import open_like_kaldi
from kaldiio.utils import parse_specifier


DEFAULT_CONF = {
    'frame_length' : 0,
    'frame_overlap': 0
}


def get_parser():
    parser = argparse.ArgumentParser(
        description='extract raw features from WAV',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default="", help="Config file")
    parser.add_argument('--write-utt2dur', type=str, help="Wspecifier to write duration of each utterance in seconds, e.g. 'ark,t:utt2dur'.")
    parser.add_argument('rspecifier', type=str, help='WAV scp file')
    parser.add_argument('wspecifier', type=str, help='Write specifier')
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    reader = kaldiio.ReadHelper(args.rspecifier)
    writer = kaldiio.WriteHelper(args.wspecifier)

    spec_dict = parse_specifier(args.write_utt2dur)
    ark_file = spec_dict['ark']
    utt2dur_writer = open_like_kaldi(ark_file, 'w')
    #utt2dur_writer = kaldiio.WriteHelper(args.write_utt2dur)

    config_file = args.config
    if config_file:
        conf = open(config_file)
        conf_dict = yaml.load(conf)
        if conf_dict is None: conf_dict = {}
        for k in DEFAULT_CONF:
            if k not in conf_dict:
                conf_dict[k] = DEFAULT_CONF[k]
        conf.close()
    else:
        conf_dict = DEFAULT_CONF

    frame_length = conf_dict['frame_length']
    frame_overlap = conf_dict['frame_overlap']

    frame_step = frame_length - frame_overlap
    #assert frame_length > 0 and frame_step > 0 and frame_step < frame_length
    for utt_id, (rate, array) in reader:
        #assert array.shape[0] >= frame_length

        if frame_length == 0:
            features = np.expand_dims(array, axis=1)
        else:
            #trailing values in array are discarded if not enough to fit in a frame
            features = []
            for frame in range(0, array.shape[0]-frame_length, frame_step):
                features.append(array[frame:frame+frame_length])

        features = np.array(features).astype(np.float32)

        writer[utt_id] = features
        utt2dur_writer.write("%s %s\n" % (utt_id, array.shape[0] / rate))
        #utt2dur_writer[utt_id] = np.array([array.shape[0] / rate])[0]

    writer.close()
    utt2dur_writer.close()


if __name__ == "__main__":
    main()
