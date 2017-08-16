from apistar import Route, Include
from apistar.docs import docs_routes
from apistar.statics import static_routes
from .controllers import detection


routes = [
    Route('/', 'POST', detection.detect_sign),
    Include('/docs', docs_routes),
    Include('/static', static_routes),
]
