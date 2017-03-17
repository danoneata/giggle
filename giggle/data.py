import os
import pdb

import matplotlib.pyplot as plt  # type: ignore

import numpy as np  # type: ignore

from pandas import (  # type: ignore
    read_sql_table,
)

from sklearn.model_selection import KFold  # type: ignore

from sqlalchemy.engine import create_engine  # type: ignore

from typing import (
    Callable,
    List,
    Tuple,
)


class Dataset:

    def __init__(self, nr_folds):
        self.nr_folds = nr_folds
        self.load()
        self.create_folds()

    def load(self) -> "Dataset":
        url = os.getenv('DATABASE_URL')
        con = create_engine(url)
        # self.data_frame = next(read_sql_table('ratings', con, chunksize=5000))
        self.data_frame = read_sql_table('ratings', con)
        return self

    def create_folds(self) -> "Dataset":
        index = np.arange(len(self.data_frame))
        kf = KFold(n_splits=self.nr_folds, shuffle=True, random_state=1337)
        self.folds = {
            i : {
                'tr': tr_index,
                'te': te_index,
            }
            for i, (tr_index, te_index) in enumerate(kf.split(index))
        }
        return self

    def load_split_fold(self, split: str, i: int) -> List[int]:
        return self.folds[i][split]

    def load_fold(self, i: int) -> Tuple[List[int], List[int]]:
        return (
            self.load_split_fold('tr', i),
            self.load_split_fold('te', i),
        )


DATASETS = {
    'large': lambda: Dataset(nr_folds=3),
}


def iqr(xs):
    q75, q25 = np.percentile(xs, [75 ,25])
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
