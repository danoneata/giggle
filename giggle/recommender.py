import pickle

import numpy as np  # type: ignore

from pandas import (  # type: ignore
    DataFrame,
)

from scipy.stats import (  # type: ignore
    norm,
)

from sklearn.metrics import mean_squared_error  # type: ignore

from typing import (
    Any,
    List,
)


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


class Recommender:

    def fit(self, data: DataFrame):
        pass

    def predict_one(self, data: DataFrame, user_id: int, joke_id: int) -> float:
        pass

    def predict_multi_jokes(self, data: DataFrame, user_id: int, joke_ids: List[int]) -> List[float]:
        pass


class GaussianRecommender(Recommender):

    def __init__(self):
        self.random_state = 1337

    def fit(self, data: DataFrame):
        self.mu = data.ratings.mean()
        self.sigma = data.ratings.std()
        return self

    def predict_rating(self, *args, **kwargs) -> float:
        return norm.rvs(
            loc=self.mu,
            scale=self.sigma,
            size=1,
            random_state=self.random_state,
        )


class BetaRecommender(Recommender):
    pass


RECOMMENDERS = {
    'gaussian': GaussianRecommender,
    'beta': BetaRecommender,
}


PATH = 'data/models/{}.pkl'


def save_recommender(key, recommender):
    with open(PATH.format(key), 'wb') as f:
        pickle.dump(recommender, f)


def load_recommender(key):
    with open(PATH.format(key), 'rb') as f:
        return pickle.load(f)
