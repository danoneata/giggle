import logging
import pdb
import os

from itertools import repeat

from flask import (  # type: ignore
    Flask,
    abort,
    jsonify,
    request,
)

from flask_sqlalchemy import SQLAlchemy  # type: ignore

from functools import partial

from logging import config as cfg

from json import dumps

from typing import (
    Tuple,
    Dict,
)

from .config import Config

from .recommender import (
    load_recommender,
)

from .models import (
    Rating,
)

from .utils import (
    wrap_exceptions,
)

app = Flask(__name__)
app.config.from_object(Config)
db = SQLAlchemy(app)

# Set-up logging
cfg.fileConfig('./config/web-service.conf')
logger = logging.getLogger('web-service')
wrap_exceptions_logger = partial(wrap_exceptions, logger=logger)


recommender = load_recommender(os.getenv('RECOMMENDER'))


@app.route('/predictInterests/<user_id>')
@wrap_exceptions_logger
def predict_interests(user_id):
    jokes = range(1, 141)
    user_joke_ids = list(zip(repeat(user_id), jokes))
    predictions = zip(recommender.predict_multi(user_joke_ids), jokes)
    predictions = sorted(predictions, reverse=True)
    predictions = [j for _, j in predictions[:5]]
    json_data = dumps(predictions, indent=4)
    logger.info(json_data)
    return json_data, 200


@app.route('/addData/', methods=['POST'])
@wrap_exceptions_logger
def add_data():
    if not request.json or 'data' not in request.json:
        return "Bad request", 400
    else:
        json_data = request.get_json('data')

    rating = Rating(
        user_id=json_data.get('user'),
        joke_id=json_data.get('joke'),
        rating=json_data.get('rating'),
    )

    db.session.add(rating)
    db.session.commit()

    json_data.update({'id': rating.id})
    return jsonify(json_data), 201
