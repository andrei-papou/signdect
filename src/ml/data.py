import csv
import os
import pickle
from typing import Iterable, Tuple, List

import numpy as np
from scipy.misc import imread, imresize

from exceptions import InvalidModeException
from ml.utils import unison_shuffle_lists
from settings import IMG_SLICE, DATA_DIRECTORY, Mode


def pad_img(img: np.array) -> np.array:
    width_pad = max(64 - img.shape[0], 0)
    height_pad = max(64 - img.shape[1], 0)
    if not width_pad and not height_pad:
        return img
    img = np.pad(img, ((0, width_pad), (0, height_pad), (0, 0)), mode='constant')
    return img


def process_conf_csv(conf_file_name: str) -> Tuple[Iterable[str], Iterable[int]]:
    image_names, labels = [], []
    file_path = os.path.expanduser(os.path.join(DATA_DIRECTORY, conf_file_name))
    with open(file_path, newline='') as csv_file:
        csv_file.readline()  # skip first line
        for row in csv.reader(csv_file):
            image_names.append(row[0])
            labels.append(int(row[1]))
    return image_names, labels


def process_image(file, img_size: int):
    img = np.array(imread(file))
    size_scalar = img_size / min(img.shape[0], img.shape[1])
    return pad_img(np.array(imresize(img, size_scalar))[IMG_SLICE]) / 255


def load_images_into_memory(directory_path: str, file_names: Iterable[str], img_size: int) -> List[np.array]:
    images = []
    container_dir = os.path.expanduser(os.path.join(DATA_DIRECTORY, directory_path))
    for file_name in file_names:
        file_path = os.path.join(container_dir, file_name)
        cache_path = '{}.{}'.format(file_path, 'cache')
        if os.path.isfile(cache_path):
            with open(cache_path, 'rb') as cache_file:
                img = pickle.load(cache_file)
        else:
            img = process_image(file_path, img_size)
            with open(cache_path, 'wb') as cache_file:
                pickle.dump(img, cache_file)
        images.append(img)
    return images


class DataProvider:

    def __init__(self, batch_size: int, img_size: int, train_data_dir: str, train_conf_file_name: str,
                 test_data_dir: str, test_conf_file_name: str):
        self.batch_size = batch_size
        self.img_size = img_size
        self.train_data_dir = train_data_dir
        self.test_data_dir = test_data_dir
        self.train_image_names, self.train_labels = process_conf_csv(train_conf_file_name)
        self.test_image_names, self.test_labels = process_conf_csv(test_conf_file_name)

    def get_batches(self, mode: Mode):
        if mode == Mode.TRAIN:
            data_dir = self.train_data_dir
            file_names = self.train_image_names
            labels = self.train_labels
        elif mode == Mode.TEST:
            data_dir = self.test_data_dir
            file_names = self.test_image_names
            labels = self.test_labels
        else:
            raise InvalidModeException
        file_names, labels = unison_shuffle_lists(file_names, labels)
        for bi in range(len(labels) // self.batch_size):
            batch_slice = slice(bi * self.batch_size, (bi + 1) * self.batch_size)
            yield load_images_into_memory(data_dir, file_names[batch_slice], self.img_size), labels[batch_slice]
        print('')


class InMemoryDataProvider(DataProvider):

    def __init__(self, batch_size: int, img_size: int, train_data_dir: str, train_conf_file_name: str,
                 test_data_dir: str, test_conf_file_name: str):
        super(InMemoryDataProvider, self)\
            .__init__(batch_size, img_size, train_data_dir, train_conf_file_name, test_data_dir, test_conf_file_name)

        train_cache_path = os.path.expanduser(os.path.join(DATA_DIRECTORY, 'train.cache'))
        if os.path.isfile(train_cache_path):
            with open(train_cache_path, 'rb') as cache_file:
                self.train_images = pickle.load(cache_file)
        else:
            self.train_images = load_images_into_memory(self.train_data_dir, self.train_image_names, self.img_size)
            with open(train_cache_path, 'wb') as cache_file:
                pickle.dump(self.train_images, cache_file)

        test_cache_path = os.path.expanduser(os.path.join(DATA_DIRECTORY, 'test.cache'))
        if os.path.isfile(test_cache_path):
            with open(test_cache_path, 'rb') as cache_file:
                self.test_images = pickle.load(cache_file)
        else:
            self.test_images = load_images_into_memory(self.test_data_dir, self.test_image_names, self.img_size)
            with open(test_cache_path, 'wb') as cache_file:
                pickle.dump(self.test_images, cache_file)

    def get_batches(self, mode: Mode):
        if mode == Mode.TRAIN:
            images = self.train_images
            labels = self.train_labels
        elif mode == Mode.TEST:
            images = self.test_images
            labels = self.test_labels
        else:
            raise InvalidModeException
        images, labels = unison_shuffle_lists(images, labels)
        num_batches = len(labels) // self.batch_size
        for bi in range(num_batches):
            batch_slice = slice(bi * self.batch_size, (bi + 1) * self.batch_size)
            yield images[batch_slice], labels[batch_slice]
            # print('Batch {} / {} produced'.format(bi, num_batches))
