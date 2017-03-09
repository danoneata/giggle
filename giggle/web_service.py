import logging
import pdb

from flask import (  # type: ignore
    Flask,
)

from functools import partial

from logging import config as cfg

from json import dumps

from .learn import (
    load_model,
)

from .utils import (
    wrap_exceptions,
)


app = Flask(__name__)


# Set-up logging
cfg.fileConfig('./config/web-service.conf')
logger = logging.getLogger('web-service')
wrap_exceptions_logger = partial(wrap_exceptions, logger=logger)


recommender = load_model()


@app.route('/predictInterests/<user_id>')
@wrap_exceptions_logger
def predict_interests(user_id):
    predictions = recommender.predict(user_id)
    predictions = predictions[:5]
    json_data = dumps(predictions, indent=4)
    logger.info(json_data)
    return json_data, 200
