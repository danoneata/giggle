import numpy as np

from sklearn.metrics import mean_squared_error

from typing import (
    Any,
    List,
)


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


class Recommender:

    def predict(self, user_id: int) -> List[Any]:
        return []


def load_model():
    return Recommender()
