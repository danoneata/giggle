import os
import pdb
import random

from collections import namedtuple

from functools import partial

from itertools import (
    count,
)

import matplotlib.pyplot as plt  # type: ignore

import numpy as np  # type: ignore

from pandas import (  # type: ignore
    DataFrame,
    read_sql_table,
)

from sklearn.model_selection import KFold  # type: ignore

from sqlalchemy.engine import create_engine  # type: ignore

from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
)


SEED = 1337


# Wrapper over data frame to include useful attributes
Data = namedtuple('Data', 'data_frame users jokes user_to_iid joke_to_iid')


class Dataset:

    def __init__(self, nr_folds: int, subsample: Optional[Callable]=None) -> None:
        self.nr_folds = nr_folds
        self.data_frame = self._load_data_frame(subsample)
        self.folds = self._get_folds()
        self.users = sorted(self.data_frame.user_id.unique())
        self.jokes = sorted(self.data_frame.joke_id.unique())
        self.user_to_iid = dict(zip(self.users, count()))
        self.joke_to_iid = dict(zip(self.jokes, count()))

    def _load_data_frame(self, subsample: Optional[Callable]) -> DataFrame:
        url = os.getenv('DATABASE_URL')
        con = create_engine(url)
        # self.data_frame = next(read_sql_table('ratings', con, chunksize=5000))
        data_frame = read_sql_table('ratings', con)
        data_frame = subsample(data_frame) if subsample else data_frame
        return data_frame

    def _get_folds(self) -> Dict[int, Dict[str, Any]]:
        index = np.arange(len(self.data_frame))
        kf = KFold(n_splits=self.nr_folds, shuffle=True, random_state=SEED)
        return {
            i: {
                'tr': tr_index,
                'te': te_index,
            }
            for i, (tr_index, te_index) in enumerate(kf.split(index))
        }

    def load_split_fold(self, split: str, i: int) -> List[int]:
        return self.folds[i][split]

    def load_fold(self, i: int) -> Tuple[List[int], List[int]]:
        return (
            self.load_split_fold('tr', i),
            self.load_split_fold('te', i),
        )

    def get_data(self, data_frame=None) -> Data:
        if data_frame is None:
            data_frame = self.data_frame
        return Data(data_frame, self.users, self.jokes, self.user_to_iid, self.joke_to_iid)


def pick_from_random_users(data_frame: DataFrame, nr_users: int) -> DataFrame:
    "Pick ratings from random users"
    random.seed(SEED)
    user_ids = random.sample(list(data_frame.user_id.unique()), nr_users)
    data_frame = data_frame[data_frame.user_id.isin(user_ids)]
    return data_frame.reset_index(drop=True, inplace=False)


pick_from_1500_random_users = lambda d: pick_from_random_users(d, nr_users=1500)


DATASETS = {
    'large': lambda: Dataset(nr_folds=3),
    'small': lambda: Dataset(nr_folds=3, subsample=pick_from_1500_random_users),
}


def data_to_user_joke_matrix(data: Data) -> np.array:
    n_users = len(data.users)
    n_jokes = len(data.jokes)
    mu = data.data_frame.rating.mean()
    user_joke_matrix = np.zeros((n_users, n_jokes)) + mu
    for _, _, u, j, r in data.data_frame.itertuples():
        user_joke_matrix[data.user_to_iid[u], data.joke_to_iid[j]] = r
    return user_joke_matrix


def iqr(xs):
    q75, q25 = np.percentile(xs, [75, 25])
    return q75 - q25


def compute_nr_bins(counts):
    # Computes the number of bins using Freedman–Diaconis' choice
    # https://en.wikipedia.org/wiki/Histogram#Number_of_bins_and_width
    diff = max(counts) - min(counts)
    width = 2 * iqr(counts) / len(counts) ** (1 / 3)
    return int(np.ceil(diff / width))


def describe_data(data):
    counts = [
        ('ratings', len(data)),
        ('users', len(data.user_id.unique())),
        ('jokes', len(data.joke_id.unique())),
    ]
    for entity, count in counts:
        print('Number of {:10s} {:7d}'.format(entity, count))


def count_ratings(data, column):
    return data.groupby([column])['id'].count()


def plot_counts(data):

    def plot_counts_(counts, xlabel, title):

        x = np.arange(len(counts)) + 1
        y = sorted(counts, reverse=True)

        plt.figure()
        plt.plot(x, y)

        plt.xlabel(xlabel)
        plt.ylabel('Counts')
        plt.title(title)

    plot_counts_(count_ratings(data, 'user_id'), 'User rank', 'Number of ratings per user')
    plot_counts_(count_ratings(data, 'joke_id'), 'Joke rank', 'Number of ratings per joke')


def plot_rating_hist(data, bins=None):
    bins = bins or compute_nr_bins(data.rating)
    outs = plt.hist(data.rating, bins=bins, normed=True)
    plt.xlabel('Rating')
    plt.ylabel('Frequency')
    return outs
