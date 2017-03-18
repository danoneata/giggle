import pdb
import pickle

from collections import defaultdict

import numpy as np  # type: ignore

from scipy.stats import (  # type: ignore
    beta,
    norm,
)

from sklearn.metrics import mean_squared_error  # type: ignore

from pandas import (
    DataFrame,
)

from typing import (
    Any,
    Iterable,
    List,
    Tuple,
)

from .data import (
    Data,
    data_to_user_joke_matrix,
)


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


class Recommender:

    def fit(self, data: Data, verbose: int):
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

    def fit(self, data: Data, verbose: int):
        self.mu = data.data_frame.rating.mean()
        self.sigma = data.data_frame.rating.std()
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

    def fit(self, data: Data, verbose: int):
        eps = 10 ** -1
        min_rating = data.data_frame.rating.min() - eps
        max_rating = data.data_frame.rating.max() + eps
        a, b, loc, scale = beta.fit(
            data.data_frame.rating,
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

    def _compute_rmse(self, data_frame: DataFrame) -> float:
        true = data_frame.rating
        pred = [self.predict(u, j) for _, u, j, _ in data_frame.itertuples()]
        return rmse(true, pred)

    def _update_params(self, data: Data) -> Iterable[None]:
        self.b_user = defaultdict(int)
        self.b_joke = defaultdict(int)
        for e in range(self.nr_epochs):
            for _, u, j, r in data.data_frame.itertuples():
                err = r - (self.mu + self.b_user[u] + self.b_joke[j])
                self.b_user[u] += self.lr * (err - self.reg * self.b_user[u])
                self.b_joke[j] += self.lr * (err - self.reg * self.b_joke[j])
                yield

    def fit(self, data: Data, verbose: int) -> Recommender:
        self.mu = data.data_frame.rating.mean()
        prev_rmse = np.inf
        TO_CHECK_PERIOD = 10000
        STOP_TOL = 1e-4
        for nr_iter, _ in enumerate(self._update_params(data)):
            if nr_iter % TO_CHECK_PERIOD == 0:
                curr_rmse = self._compute_rmse(data.data_frame)
                if verbose:
                    print('{:5.0f} {:.2f}'.format(nr_iter / TO_CHECK_PERIOD, curr_rmse))
                if np.abs(curr_rmse - prev_rmse) / curr_rmse < STOP_TOL:
                    break
                else:
                    prev_rmse = curr_rmse
        return self

    def predict(self, user_id: int, joke_id: int) -> float:
        return self.mu + self.b_user[user_id] + self.b_joke[joke_id]


class Neighbourhood(Recommender):

    def __init__(self, k: int) -> None:
        self.k = k

    def _find_most_similar_jokes(self, joke_id: int) -> List[int]:
        i = self.data.joke_to_iid[joke_id]
        joke_iids = np.argsort(-self.sims[i])
        joke_iids = joke_iids[1: self.k + 1]
        return joke_iids

    def fit(self, data: Data, verbose: int) -> Recommender:
        self.user_joke_matrix = data_to_user_joke_matrix(data)
        self.sims = np.corrcoef(self.user_joke_matrix.T)
        self.data = data
        return self

    def predict(self, user_id: int, joke_id: int) -> float:
        user_iid = self.data.user_to_iid[user_id]
        joke_iid = self.data.joke_to_iid[joke_id]
        jokes = self._find_most_similar_jokes(joke_id)
        sum_rat = sum(self.sims[j, joke_iid] * self.user_joke_matrix[user_iid, j] for j in jokes)
        sum_sim = sum(self.sims[j, joke_iid] for j in jokes)
        return sum_rat / sum_sim


RECOMMENDERS = {
    'gaussian': GaussianRecommender(),
    'beta': BetaRecommender(),
    'baseline': BaselineRecommender(nr_epochs=10, lr=0.01, reg=0.1),
    'neigh': Neighbourhood(k=10),
    # 'neigh_mean': Neighbourhood(),
    # 'neigh_base': Neighbourhood(),
}


def get_recommender_path(key: str) -> str:
    PATH = 'data/models/{}.pkl'
    return PATH.format(key)


def save_recommender(path: str, recommender: Recommender):
    with open(path, 'wb') as f:
        pickle.dump(recommender, f)


def load_recommender(path: str) -> Recommender:
    with open(path, 'rb') as f:
        return pickle.load(f)
