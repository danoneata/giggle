import pdb
import pickle

from collections import defaultdict

import numpy as np  # type: ignore

from pandas import (  # type: ignore
    DataFrame,
)

from scipy.stats import (  # type: ignore
    beta,
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

    def __init__(self):
        self.random_state = 1337

    def fit(self, data: DataFrame):
        eps = 10 ** -1
        min_rating = data.rating.min() - eps
        max_rating = data.rating.max() + eps
        a, b, loc, scale = beta.fit(
            data.rating,
            floc=min_rating,
            fscale=max_rating - min_rating,
        )
        self.a = a
        self.b = b
        self.loc = loc
        self.scale = scale
        return self

    def predict(self, user_id: int, joke_id: int) -> float:
        value, = beta.rvs(
            a=self.a,
            b=self.b,
            loc=self.loc,
            scale=self.scale,
            size=1,
            random_state=self.random_state,
        )
        return value

    def predict_multi(self, user_joke_ids: List[Tuple[int, int]]) -> List[float]:
        size = len(user_joke_ids)
        return beta.rvs(
            a=self.a,
            b=self.b,
            loc=self.loc,
            scale=self.scale,
            size=size,
            random_state=self.random_state,
        )


class BaselineRecommender(Recommender):

    def __init__(self, nr_epochs, lr, reg):
        self.nr_epochs = nr_epochs
        self.reg = reg
        self.lr = lr
        self.mu = None
        self.b_user = None
        self.b_joke = None

    def fit(self, data: DataFrame) -> Recommender:
        self.mu = data.rating.mean()
        nr_users = len(data.user_id.unique())
        nr_jokes = len(data.joke_id.unique())
        self.b_user = defaultdict(int)
        self.b_joke = defaultdict(int)
        for e in range(self.nr_epochs):
            for _, i, u, j, r in data.itertuples():
                err = r - (self.mu + self.b_user[u] + self.b_joke[j])
                self.b_user[u] += self.lr * (err - self.reg * self.b_user[u])
                self.b_joke[j] += self.lr * (err - self.reg * self.b_joke[j])
        return self

    def predict(self, user_id: int, joke_id: int) -> float:
        return self.mu + self.b_user[user_id] + self.b_joke[joke_id]


RECOMMENDERS = {
    'gaussian': GaussianRecommender(),
    'beta': BetaRecommender(),
    'baseline': BaselineRecommender(5, 0.01, 0.1),
}


PATH = 'data/models/{}.pkl'


def save_recommender(key: str, recommender: Recommender):
    with open(PATH.format(key), 'wb') as f:
        pickle.dump(recommender, f)


def load_recommender(key: str) -> Recommender:
    with open(PATH.format(key), 'rb') as f:
        return pickle.load(f)
