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

import numpy as np  # type: ignore

from typing import (
    Dict,
    List,
    Tuple,
)

from .config import Config

from .recommender import (
    get_recommender_path,
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


recommender_key = os.getenv('RECOMMENDER')
recommender = load_recommender(get_recommender_path(recommender_key))


def get_unrated_jokes(user_id: int) -> List[int]:
    jokes = set(joke_id for joke_id, in Rating.query.with_entities(Rating.joke_id).distinct(Rating.joke_id))
    jokes_rated = set(joke_id for joke_id, in Rating.query.with_entities(Rating.joke_id).filter_by(user_id=user_id))
    return list(jokes - jokes_rated)


@app.route('/predictInterests/<user_id>')
@wrap_exceptions_logger
def predict_interests(user_id):
    user_id = int(user_id)
    jokes = get_unrated_jokes(user_id)
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
    json_data = request.get_json()

    if not json_data:
        return jsonify("Bad request"), 400

    rating = Rating(
        user_id=json_data.get('user'),
        joke_id=json_data.get('joke'),
        rating=json_data.get('rating'),
    )

    db.session.add(rating)
    db.session.commit()

    return jsonify(json_data), 201


@app.route('/similarItems/<joke_id>')
@wrap_exceptions_logger
def similar_items(joke_id):

    if not recommender_key.startswith('neigh'):
        return jsonify("End-point works only with neighbourhood-based method"), 400

    joke_to_iid = recommender.data.joke_to_iid
    iid_to_joke = {i: j for j, i in joke_to_iid.items()}

    joke_id = int(joke_id)
    joke_iid = joke_to_iid[joke_id]

    iids = np.argsort(-recommender.sims[joke_iid])

    json_data = [
        {
            'id': int(iid_to_joke[iid]),
            'similarity': float(recommender.sims[joke_iid, iid]),
        }
        for iid in iids
        if iid != joke_iid
    ]
    json_data = json_data[:5]
    return jsonify(json_data), 200
