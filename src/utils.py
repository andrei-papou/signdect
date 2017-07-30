import os
import random
from time import time
from typing import Iterable, List
import numpy as np
from settings import MODEL_DIRECTORY, Mode
from exceptions import InvalidModeException


def unison_shuffle(arrays: Iterable[Iterable], perm_length: int) -> List[np.array]:
    permutation = np.random.permutation(perm_length)
    return [np.array(a)[permutation] for a in arrays]


def unison_shuffle_lists(*lists):
    zipped_lists = list(zip(*lists))
    random.shuffle(zipped_lists)
    return zip(*zipped_lists)


def generate_model_id():
    return str(int(time()))


def get_model_file_path(model_id: int) -> str:
    return os.path.join(MODEL_DIRECTORY, 'model_{}.ckpt'.format(model_id))


def get_mode_prefix(mode: Mode) -> str:
    if mode == Mode.TRAIN:
        return 'Training'
    elif mode == Mode.TEST:
        return 'Validation'
    else:
        raise InvalidModeException
