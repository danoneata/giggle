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
    FOLDS,
    load_fold,
)

from .recommender import Recommender


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def evaluate_fold(i: int, data: DataFrame, recommender: Recommender, verbose: int=0) -> float:
    if verbose: print('-- Fold', i)
    tr_idxs, te_idxs = load_fold(i)
    tr_data, te_data = data.ix[tr_idxs], data.ix[te_idxs]
    recommender.fit(tr_data)
    true = te_data.rating
    pred = recommender.predict_multi(te_data[['user_id', 'joke_id']].values)
    return rmse(true, pred)


def evaluate_folds(data: DataFrame, recommender: Recommender, verbose: int=0) -> Dict[int, float]:
    return {
        i: evaluate_fold(i, data, recommender, verbose)
        for i in FOLDS
    }


def print_results(results: Dict[int, float]):
    values = list(results.values())
    nfolds = len(FOLDS)
    for i in FOLDS:
        print("{:2d} {:.4f}".format(i, results[i]))
    print("--------------")
    print("{:.4f} ± {:.3f}".format(
        np.mean(values),
        np.std(values) / np.sqrt(nfolds)
    ))
