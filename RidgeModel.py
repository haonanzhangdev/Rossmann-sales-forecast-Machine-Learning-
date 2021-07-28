import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from MyMetrics import rmspe_origin,get_rmspe


def get_best_ridge(X_train_std, y_train,X_test_std,y_test):
    # One-hot encoding of "DayOfWeek" columns
    # X = pd.DataFrame(X)
    # index = X['DayOfWeek']
    #
    # L = [[0 if index.iloc[j] != i else 1 for i in range(7)] for j in range(index.shape[0])]
    # L = pd.DataFrame(L)
    # X = pd.concat((X, L), axis=1)
    # del X['DayOfWeek']
    # del X['Day']

    # use gridsearchCV to find the best alpha params automatically
    alphas = np.logspace(-3, 2, 50)
    gridsearch = GridSearchCV(Ridge(), param_grid={'alpha': alphas}, cv=5, scoring=get_rmspe(),verbose=1)
    gridsearch.fit(X_train_std, y_train)
    changetype = list(gridsearch.best_params_.values())

    # create the ridge regression model
    ridge = Ridge(alpha=changetype[0], fit_intercept=True, max_iter=2000, normalize=True, solver='lsqr', tol=0.1)

    print('Best Ridge parameters:\n', ridge)

    # train the ridge model
    ridge = ridge.fit(X_train_std, y_train)

    print('Train set score:', rmspe_origin(y_train, ridge.predict(X_train_std)))
    print('Test set score：', rmspe_origin(y_test, ridge.predict(X_test_std)))

    return Ridge(alpha=changetype[0], fit_intercept=True, max_iter=2000, normalize=True, solver='lsqr', tol=0.1)



if __name__ == '__main__':
    # read data
    train = pd.read_csv('train.csv', low_memory=False, index_col='Date')
    test = pd.read_csv('test.csv', low_memory=False, index_col='Date')
    store = pd.read_csv('store.csv', low_memory=False)

    from DataCleaning import clean_merge_data, clean_store

    # clean data and merge them together
    store_cleaned = clean_store(store)
    X, y = clean_merge_data(train, store_cleaned, True)

    # split train set to train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=233)

    # normalize the data
    std_scaler = StandardScaler()
    std_scaler.fit(X_train)
    X_train_std = std_scaler.transform(X_train)
    X_test_std = std_scaler.transform(X_test)

    ridge = get_best_ridge(X_train_std, y_train,X_test_std,y_test)

    # # predicted and get the score
    # predicted = ridge.predict(X_test_std)
    # score = rmspe_origin(y_test, predicted)
    #
    # print(predicted)
    # print(score)
