from collections import namedtuple
from sqlalchemy import Column, Integer, String, BigInteger
from sqlalchemy.orm import relationship
from web.database import Base


Permission = namedtuple('Permission', ['code', 'name'])
permissions = [
    Permission(code=0b00000000000001, name='Upload signs'),
    Permission(code=0b00000000000010, name='View signs list'),
    Permission(code=0b00000000000100, name='View sign detail'),
    Permission(code=0b00000000001000, name='View admin panel'),
    Permission(code=0b00000000010000, name='Assign roles'),
    Permission(code=0b00000000100000, name='Update roles'),
]


class Role(Base):
    __tablename__ = 'roles'

    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False, unique=True)
    permissions = Column(BigInteger, nullable=False)


class User(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True)
    email = Column(String(255), nullable=False, unique=True)
    password = Column(String(512), nullable=False)
    first_name = Column(String(127))
    last_name = Column(String(127))
    role = relationship('Role', 'users', dynamic=True)
