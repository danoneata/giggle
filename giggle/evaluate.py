import pdb

import numpy as np  # type: ignore

from pandas import (  # type: ignore
    DataFrame,
)

from typing import (
    Any,
    Dict,
    Callable,
)

from .data import (
    Data,
    Dataset,
)

from .recommender import (
    Recommender,
    rmse,
)


def evaluate_fold(i: int, dataset: Dataset, recommender: Recommender, verbose: int=0) -> float:
    if verbose:
        print('-- Fold', i)
    tr_idxs, te_idxs = dataset.load_fold(i)
    tr_data = dataset.get_data(dataset.data_frame.ix[tr_idxs])
    te_data = dataset.get_data(dataset.data_frame.ix[te_idxs])
    recommender.fit(tr_data, verbose)
    true = te_data.data_frame.rating
    pred = recommender.predict_multi(te_data.data_frame[['user_id', 'joke_id']].values)
    return rmse(true, pred)


def evaluate_folds(dataset: Dataset, recommender: Recommender, verbose: int=0) -> Dict[int, float]:
    return {
        i: evaluate_fold(i, dataset, recommender, verbose)
        for i in range(dataset.nr_folds)
    }


def print_results(results: Dict[int, float]):
    values = list(results.values())
    nfolds = len(results)
    for i in range(nfolds):
        print("{:2d} {:.4f}".format(i, results[i]))
    print("--------------")
    print("{:.4f} ± {:.3f}".format(
        np.mean(values),
        np.std(values) / np.sqrt(nfolds)
    ))
