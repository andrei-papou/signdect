from enum import Enum
from web.database import Base


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


API_STAR_SETTINGS = {
    'SECRET_KEY': '^7s7s8721%S%&gggd121%^&^&^*&Ysfgsfg1212312323213',
    'DATABASE': {
        'URL': 'postgresql://signs_admin:homm1994@localhost:5432/signs',
        'METADATA': Base.metadata
    },
    'MODEL_PARAMS': {
        'width': WIDTH,
        'height': HEIGHT,
        'channels': CHANNELS,
        'categories': CATEGORIES,
        'saved_model': BEST_MODEL_ID
    }
}

