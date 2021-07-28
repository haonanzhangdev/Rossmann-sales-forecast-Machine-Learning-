import warnings

from sklearn.preprocessing import StandardScaler

from MyMetrics import rmspe_origin
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split
import numpy as np
import pandas as pd
from DataCleaning import clean_store, clean_merge_data

warnings.filterwarnings("ignore")


def get_best_RandomForestRegressor(X_train_std, y_train, X_test_std, y_test):
    # ======== Adjustment Parameters=======================
    # max_features = ['auto']
    # max_depth = [int(x) for x in np.linspace(10, 20, num = 2)]  # 从10开始，到20结束，步长为2，可按照需要修改数值
    # max_depth.append(None)
    #
    # random_grid = {
    #                # 'n_estimators': n_estimators,
    #                'max_features': max_features,
    #                'max_depth': max_depth}
    #
    # randforest = RandomForestRegressor(n_estimators=100)
    # rfr1_random = RandomizedSearchCV(randforest, param_distributions=random_grid,
    #                               n_iter = 50, scoring='neg_mean_absolute_error',
    #                               cv = 3, verbose=2, random_state=42, n_jobs= -1)
    # rfr1_random.fit(X_train_std, y_train)
    # print(rfr1_random.best_params_)
    # best_random_model = rfr1_random.best_estimator_

    # create the RandomForestRegressor model
    randforest = RandomForestRegressor(n_estimators=80,max_depth=650,min_samples_split=300000,
                                       min_samples_leaf=3,max_features=0.5,max_samples=0.7,verbose=2)
    print('best random forest parameters:\n', randforest)

    randforestreg = randforest.fit(X_train_std, y_train)
    print('Train set score：', rmspe_origin(y_train, randforestreg.predict(X_train_std)))
    print('Test set score：', rmspe_origin(y_test, randforestreg.predict(X_test_std)))

    return randforestreg


if __name__ == '__main__':
    # read data
    train = pd.read_csv('train.csv', low_memory=False, index_col='Date')
    test = pd.read_csv('test.csv', low_memory=False, index_col='Date')
    store = pd.read_csv('store.csv', low_memory=False)

    # clean data and merge them together
    store_cleaned = clean_store(store)
    X, y = clean_merge_data(train, store_cleaned, True)
    real_test = clean_merge_data(test, store_cleaned, False)
    # split train set to train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=233)

    # normalize the data
    std_scaler = StandardScaler()
    std_scaler.fit(X_train)
    X_train_std = std_scaler.transform(X_train)
    X_test_std = std_scaler.transform(X_test)
    real_test = std_scaler.transform(real_test)

    best_model = get_best_RandomForestRegressor(X_train_std, y_train, X_test_std, y_test)

    # predict the real test data and save it into submission.csv
    real_pred = best_model.predict(real_test)

    submission = pd.DataFrame({'Id':test['Id'].values})
    submission['Sales'] = real_pred
    print(submission)
    # another thing need to do is assign all cases whose Open == 0 to 0
    index = (test['Open'] == 0).values
    submission.loc[index, 'Sales'] = 0

    # save to submission.csv
    submission.to_csv('submission.csv', sep=',', index=False)