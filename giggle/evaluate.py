import pdb

import matplotlib.pyplot as plt  # type: ignore

import numpy as np  # type: ignore

from pandas import (  # type: ignore
    DataFrame,
)

from typing import (
    Any,
    Callable,
    Dict,
    List,
    Tuple,
)

from .data import (
    Data,
    Dataset,
)

from .recommender import (
    Recommender,
    rmse,
)


def evaluate_fold(i: int, dataset: Dataset, recommender: Recommender, verbose: int=0) -> Tuple[float, Any]:
    if verbose:
        print('-- Fold', i)
    tr_idxs, te_idxs = dataset.load_fold(i)
    tr_data = dataset.get_data(dataset.data_frame.ix[tr_idxs])
    te_data = dataset.get_data(dataset.data_frame.ix[te_idxs])
    recommender.fit(tr_data, verbose)
    true = te_data.data_frame.rating.values
    pred = recommender.predict_multi(te_data.data_frame[['user_id', 'joke_id']].values)
    return rmse(true, pred), (true, pred)


def evaluate_folds(dataset: Dataset, recommender: Recommender, verbose: int=0) -> Tuple[Tuple[float], Tuple[Any]]:
    return tuple(zip(*[
        evaluate_fold(i, dataset, recommender, verbose)
        for i in range(dataset.nr_folds)
    ]))


def print_results(results: List[float]):
    for i, res in enumerate(results, start=1):
        print("{:2d} {:.4f}".format(i, res))
    print("--------------")
    print("{:.4f} ± {:.3f}".format(np.mean(results), np.std(results) / len(results)))


def scatter_plot(list_of_data, path):
    true = np.hstack([data[0] for data in list_of_data])
    pred = np.hstack([data[1] for data in list_of_data])
    true = true[::1000]
    pred = pred[::1000]
    plt.scatter(true, pred)
    plt.xlabel('True ratings')
    plt.ylabel('Estimated ratings')
    plt.savefig(path)
