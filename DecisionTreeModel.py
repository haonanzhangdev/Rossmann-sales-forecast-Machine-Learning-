import warnings
import pandas as pd

from sklearn.tree import DecisionTreeRegressor
from DataCleaning import clean_merge_data, clean_store
from MyMetrics import rmspe_origin
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


def get_best_DecisionTreeRegressor(X_train_std, y_train, X_test_std, y_test):
    ##---------------Decision Tree-----------------------
    DT = DecisionTreeRegressor(min_samples_leaf=20, random_state=0)
    DTmodel = DT.fit(X_train_std, y_train)
    y_pred = DTmodel.predict(X_test_std)

    print('Best Decision Tree parameters:\n', DT)
    print('Train set score:', rmspe_origin(y_train, DTmodel.predict(X_train_std)))
    print('Test set score：', rmspe_origin(y_test, y_pred))

    # print("Predictions saved.")
    return DecisionTreeRegressor(min_samples_leaf=20, random_state=0)


if __name__ == '__main__':
    # read data
    train = pd.read_csv('train.csv', low_memory=False, index_col='Date')
    test = pd.read_csv('test.csv', low_memory=False, index_col='Date')
    store = pd.read_csv('store.csv', low_memory=False)

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

    get_best_DecisionTreeRegressor(X_train_std, y_train, X_test_std, y_test)
