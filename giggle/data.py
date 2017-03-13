import os
import pdb

import matplotlib.pyplot as plt

import numpy as np

from pandas import (
    read_sql_table,
)

from sqlalchemy.engine import create_engine


def load_data():
    url = os.getenv('DATABASE_URL')
    con = create_engine(url)
    # return next(read_sql_table('ratings', con, chunksize=5000))
    return read_sql_table('ratings', con)


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
