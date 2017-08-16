from apistar.http import Response, RequestData
from apistar.backends import SQLAlchemy
from psycopg2 import connect
from ml.data import process_image
from web.providers import MlModel


def detect_sign(data: RequestData, model: MlModel) -> Response:
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