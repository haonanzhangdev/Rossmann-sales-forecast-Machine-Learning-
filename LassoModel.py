import math
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, Lasso
from sklearn.preprocessing import StandardScaler
from DataCleaning import clean_store, clean_merge_data
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
from MyMetrics import get_rmspe, rmspe_origin

warnings.filterwarnings("ignore")


def get_best_lasso(train_x, train_y, test_x, test_y):
    param_grid = [
        {
            'alpha': [1e-20, 0.1, 0.5],
            'normalize': [True],
            'selection': ['random'],
            'positive': [False],
            'warm_start': [False]
        }
    ]

    score = get_rmspe()
    lasso_grid = Lasso()
    grid_search = GridSearchCV(lasso_grid, param_grid, cv=3, scoring=score, verbose=1)

    grid_search.fit(train_x, train_y)
    # print(grid_search.best_params_)
    # print(grid_search.best_estimator_)

    print('train set score:', rmspe_origin(train_y, grid_search.predict(train_x)))
    print('test set score：', rmspe_origin(test_y, grid_search.predict(test_x)))

    best_lasso_grid = grid_search.best_estimator_  # find the best model with corresponding alpha
    print('best lasso paramters:', best_lasso_grid)
    return best_lasso_grid


if __name__ == '__main__':
    train = pd.read_csv('train.csv', low_memory=False, index_col='Date')
    test = pd.read_csv('test.csv', low_memory=False, index_col='Date')
    store = pd.read_csv('store.csv', low_memory=False)

    store_cleaned = clean_store(store)
    # print(store_cleaned.info())
    # print(store_cleaned.columns)

    X, y = clean_merge_data(train, store_cleaned, True)
    X = X[y < 12000]
    y = y[y < 12000]
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.25, random_state=233)

    std_scaler = StandardScaler()
    std_scaler.fit(train_x, train_y)
    train_x_std = std_scaler.transform(train_x)
    test_x_std = std_scaler.transform(test_x)

    best_lasso = get_best_lasso(train_x_std, train_y, test_x_std, test_y)

    # best_lasso = Lasso(alpha=0, normalize=True, selection='random', positive=False, warm_start=False)
    # best_lasso.fit(train_x_std, train_y)
    # y_pred = best_lasso.predict(test_x_std)
    # print('test,rmspe:', rmspe_origin(test_y, y_pred))
