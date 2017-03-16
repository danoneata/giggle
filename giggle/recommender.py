import pickle

import numpy as np  # type: ignore

from pandas import (  # type: ignore
    DataFrame,
)

from scipy.stats import (  # type: ignore
    norm,
)

from typing import (
    Any,
    List,
    Tuple,
)


class Recommender:

    def fit(self, data: DataFrame):
        pass

    def predict(self, user_id: int, joke_id: int) -> float:
        pass

    def predict_multi(self, user_joke_ids: List[Tuple[int, int]]) -> List[float]:
        return [
            self.predict(user_id, joke_id)
            for user_id, joke_id in user_joke_ids
        ]


class GaussianRecommender(Recommender):

    def __init__(self):
        self.random_state = 1337

    def fit(self, data: DataFrame):
        self.mu = data.rating.mean()
        self.sigma = data.rating.std()
        return self

    def predict(self, user_id: int, joke_id: int) -> float:
        value, = norm.rvs(
            loc=self.mu,
            scale=self.sigma,
            size=1,
            random_state=self.random_state,
        )
        return value

    def predict_multi(self, user_joke_ids: List[Tuple[int, int]]) -> List[float]:
        size = len(user_joke_ids)
        return norm.rvs(
            loc=self.mu,
            scale=self.sigma,
            size=size,
            random_state=self.random_state,
        )


class BetaRecommender(Recommender):
    pass


RECOMMENDERS = {
    'gaussian': GaussianRecommender,
    'beta': BetaRecommender,
}


PATH = 'data/models/{}.pkl'


def save_recommender(key: str, recommender: Recommender):
    with open(PATH.format(key), 'wb') as f:
        pickle.dump(recommender, f)


def load_recommender(key: str) -> Recommender:
    with open(PATH.format(key), 'rb') as f:
        return pickle.load(f)
