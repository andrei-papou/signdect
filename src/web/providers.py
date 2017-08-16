import jwt
from jwt.exceptions import DecodeError, ExpiredSignatureError
from apistar.http import Header
from apistar.settings import Settings
from apistar.exceptions import ValidationError
from ml.model import Model
from web.models.auth import User


class MlModel:
    model = None

    @classmethod
    def build(cls, settings: Settings):
        if cls.model is None:
            cls.model = Model(**settings['MODEL_PARAMS'])
        return cls.model


class AuthUser:

    @classmethod
    def build(cls, authorization: Header, settings: Settings):
        try:
            data = jwt.decode(authorization, key=settings['SECRET_KEY'], algorithms=[settings['HASHING_ALG']])
        except DecodeError:
            raise ValidationError(detail={'error': 'Invalid toke'})
        except ExpiredSignatureError:
            raise ValidationError(detail={'error': 'Token is expired'})
