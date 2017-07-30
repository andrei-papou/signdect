from enum import Enum


WIDTH = 64
HEIGHT = 64
CHANNELS = 3
CATEGORIES = 106

IMG_SLICE = (slice(0, 64), slice(0, 64))
MODEL_DIRECTORY = '/tmp/signdect'
DATA_DIRECTORY = '~/data/signdect'

BEST_MODEL_ID = '1501397061'


class Mode(Enum):
    TRAIN = 1
    TEST = 2


DATABASE_CONF = {
    'database': 'signs',
    'user': 'signs_admin',
    'password': 'homm1994',
    'host': 'localhost',
    'port': '5432'
}
