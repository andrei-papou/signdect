from apistar import App
from web.routes import routes
from settings import API_STAR_SETTINGS


app = App(routes=routes, settings=API_STAR_SETTINGS)
