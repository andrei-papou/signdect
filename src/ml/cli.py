import argparse
import sys

import numpy as np
import tensorflow as tf
from .data import DataProvider

from ml.model import Model
from settings import WIDTH, HEIGHT, CHANNELS, CATEGORIES

tf.logging.set_verbosity(tf.logging.INFO)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_id', type=str, help='Postfix for model filename')
    parser.add_argument('--saved_model', type=str, help='Id of the model to start training with')

    return parser.parse_args(argv)


def main(args):
    np.random.seed()

    data_provider = DataProvider(
        batch_size=50,
        img_size=64,
        train_data_dir='train',
        train_conf_file_name='gt_train.csv',
        test_data_dir='test',
        test_conf_file_name='gt_test.csv')
    model = Model(
        width=WIDTH,
        height=HEIGHT,
        channels=CHANNELS,
        categories=CATEGORIES,
        stop_in=10,
        model_id=args.model_id,
        saved_model=args.saved_model)
    model.train(
        data_provider=data_provider,
        num_epochs=200,
        mini_batch_size=50,
        learning_rate=0.0001)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
