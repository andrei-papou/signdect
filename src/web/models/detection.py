from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import relationship
from web.database import Base


class Category(Base):
    __tablename__ = 'categories'

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)

    def __repr__(self):
        return 'Category(name={})'.format(self.name)

    def __str__(self):
        return '<Category: {}>'.format(self.name)


class Sign(Base):
    __tablename__ = 'signs'

    id = Column(Integer, primary_key=True)
    app_code = Column(Integer, nullable=False, unique=True)
    pdd_code = Column(String(32), nullable=False, unique=True)
    name = Column(String(256), nullable=False)
    img_url = Column(String(512), nullable=False)
    category = relationship('Category', backref='signs', lazy='dynamic')

    def __repr__(self):
        return 'Sign(app_code={}, pdd_code={}, name={}, img_url={})'\
            .format(self.app_code, self.pdd_code, self.name, self.img_url)

    def __str__(self):
        return '<Sign: {} ({})>'.format(self.pdd_code, self.name)
