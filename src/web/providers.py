from apistar.http import Header
from apistar.settings import Settings
from ml.model import Model
from settings import WIDTH, HEIGHT, CHANNELS, CATEGORIES, BEST_MODEL_ID


model = Model(width=WIDTH, height=HEIGHT, channels=CHANNELS, categories=CATEGORIES, saved_model=BEST_MODEL_ID)


class MlModel:
    model = None

    @classmethod
    def build(cls, settings: Settings):
        if cls.model is None:
            cls.model = Model(**settings['MODEL_PARAMS'])
        return cls.model


class Authentication:

    @classmethod
    def build(cls, authorization: Header, settings: Settings):
        pass
