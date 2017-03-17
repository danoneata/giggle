import pdb

import numpy as np  # type: ignore

from pandas import (  # type: ignore
    DataFrame,
)

from sklearn.metrics import mean_squared_error  # type: ignore

from typing import (
    Any,
    Dict,
    Callable,
)

from .data import (
    Dataset,
)

from .recommender import Recommender


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def evaluate_fold(i: int, dataset: Dataset, recommender: Recommender, verbose: int=0) -> float:
    if verbose: print('-- Fold', i)
    tr_idxs, te_idxs = dataset.load_fold(i)
    tr_data, te_data = dataset.data_frame.ix[tr_idxs], dataset.data_frame.ix[te_idxs]
    recommender.fit(tr_data)
    true = te_data.rating
    pred = recommender.predict_multi(te_data[['user_id', 'joke_id']].values)
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
