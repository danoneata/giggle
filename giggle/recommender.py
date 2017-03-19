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

    def _compute_similarities(self, user_joke_matrix: np.array) -> np.array:
        MIN_SUPPORT = 5
        _, nr_jokes = user_joke_matrix.shape
        joke_joke_matrix = np.zeros((nr_jokes, nr_jokes))
        np.fill_diagonal(joke_joke_matrix, 1)
        for i in range(nr_jokes):
            for j in range(i + 1, nr_jokes):
                common = np.logical_and(
                    np.logical_not(np.isnan(user_joke_matrix[:, i])),
                    np.logical_not(np.isnan(user_joke_matrix[:, j])),
                )
                if np.sum(common) < MIN_SUPPORT:
                    continue
                r_i = user_joke_matrix[common, i] - user_joke_matrix[common, i].mean()
                r_j = user_joke_matrix[common, j] - user_joke_matrix[common, j].mean()
                numer = np.sum(r_i * r_j)
                denom = np.sqrt(np.sum(r_i ** 2)) * np.sqrt(np.sum(r_j ** 2))
                joke_joke_matrix[i, j] = joke_joke_matrix[j, i] = numer / denom
        return joke_joke_matrix

    def _find_most_similar_rated_jokes(self, user_ratings: np.array, joke_id: int) -> List[int]:
        i = self.data.joke_to_iid[joke_id]
        rated_iids, = np.where(np.logical_not(np.isnan(user_ratings)))
        joke_iids = np.argsort(-self.sims[i])
        joke_iids = [iid for iid in joke_iids if iid != i and iid in rated_iids]
        joke_iids = joke_iids[:self.k]
        return joke_iids

    def fit(self, data: Data, verbose: int) -> Recommender:
        self.user_joke_matrix = data_to_user_joke_matrix(data)
        self.sims = self._compute_similarities(self.user_joke_matrix)
        self.data = data
        self.mu = data.data_frame.rating.mean()
        return self

    def predict(self, user_id: int, joke_id: int) -> float:
        user_iid = self.data.user_to_iid[user_id]
        joke_iid = self.data.joke_to_iid[joke_id]
        user_ratings = self.user_joke_matrix[user_iid]
        jokes = self._find_most_similar_rated_jokes(user_ratings, joke_id)
        sum_rat = np.sum(self.sims[joke_iid, jokes] * user_ratings[jokes])
        sum_sim = np.sum(self.sims[joke_iid, jokes])
        return (sum_rat / sum_sim) if sum_sim != 0 else self.mu


RECOMMENDERS = {
    'gaussian': GaussianRecommender(),
    'beta': BetaRecommender(),
    'baseline': BaselineRecommender(nr_epochs=10, lr=0.01, reg=0.1),
    'neigh': Neighbourhood(k=35),
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
