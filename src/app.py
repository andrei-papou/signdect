from psycopg2 import connect
from apistar import App, Route, Include, http, Response
from apistar.docs import docs_routes
from apistar.statics import static_routes
from data import process_image
from model import Model
from settings import WIDTH, HEIGHT, CHANNELS, CATEGORIES, BEST_MODEL_ID, DATABASE_CONF


model = Model(width=WIDTH, height=HEIGHT, channels=CHANNELS, categories=CATEGORIES, saved_model=BEST_MODEL_ID)


def detect_sign(data: http.RequestData) -> Response:
    file = next(data.values())
    img = process_image(file, 64)
    category = int(model.infer(img))

    with connect(**DATABASE_CONF) as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT id, name, pdd_code, img_url FROM signs WHERE app_code = %s', (category,))
        result = cursor.fetchone()

    if result is None:
        data = {'message': 'Sorry, cannot detect your sign.'}
        return Response(data=data, status=404, headers={'Access-Control-Allow-Origin': '*'})

    s_id, name, pdd_code, img_url = result
    data = {
        'id': s_id,
        'name': name,
        'pdd_code': pdd_code,
        'img_url': img_url
    }
    return Response(data=data, status=200, headers={'Access-Control-Allow-Origin': '*'})


routes = [
    Route('/', 'POST', detect_sign),
    Include('/docs', docs_routes),
    Include('/static', static_routes)
]


app = App(routes=routes)
