import warnings
import pandas as pd
from pandas import DataFrame

warnings.filterwarnings("ignore")


def clean_store(store: DataFrame):
    '''
    this function is used to clean store data
    return: cleaned store dataframe
    '''
    # fill the missing values with median
    store['CompetitionDistance'].fillna(store['CompetitionDistance'].median(), inplace=True)

    # map the non numeric values into intergers
    store['StoreType'] = store['StoreType'].map({'a': 0, 'b': 1, 'c': 2, 'd': 3})
    store['Assortment'] = store['Assortment'].map({'a': 0, 'b': 1, 'c': 2})

    # fill the missing values with 0
    store.fillna(0, inplace=True)

    # drop last column
    store.drop(labels='PromoInterval', axis=1, inplace=True)

    return store


def clean_merge_data(data: DataFrame, store: DataFrame, istraindata: bool):
    """
    this function is used to merge train|test and store toghther
    params: istraindata: bool, if yes, then data is from train.csv else data is from test.csv
    return: trainX,trainy or testX,testy
    """
    data.index = pd.DatetimeIndex(data.index)
    # remove bias data
    if istraindata:
        data = data[(data['Sales'] != 0) & (data['Open'] != 0)]
    mapping = {0: 0, '0': 0, 'a': 1, 'b': 2, 'c': 3}
    data.StateHoliday.replace(mapping, inplace=True)

    # create new features from date
    data['Year'] = data.index.year
    data['Month'] = data.index.month
    data['Day'] = data.index.day
    data['WeekOfYear'] = data.index.weekofyear

    data_store = pd.merge(data, store, how='inner', on='Store')

    # create new features through CompetitionOpenSinceMonth CompetitionOpenSinceYear Promo2SinceWeek Promo2SinceYear
    data_store['CompetitionOpen'] = 12 * (data_store.Year - data_store.CompetitionOpenSinceYear) + (
                data_store.Month - data_store.CompetitionOpenSinceMonth)
    data_store['PromoOpen'] = 12 * (data_store.Year - data_store.Promo2SinceYear) + (
                data_store.WeekOfYear - data_store.Promo2SinceWeek) / 4.0
    data_store.fillna(0, inplace=True)

    # check whether it is train data or test data
    if istraindata:
        data_store_X = data_store.drop(labels=['Store', 'Sales',
                                               'Customers',
                                               'CompetitionOpenSinceMonth',
                                               'CompetitionOpenSinceYear',
                                               'Promo2SinceWeek',
                                               'Promo2SinceYear',
                                               'WeekOfYear'], axis=1)
        data_store_y = data_store['Sales']
        return data_store_X, data_store_y

    else:
        data_store_X = data_store.drop(labels=['Id', 'Store',
                                               'CompetitionOpenSinceMonth',
                                               'CompetitionOpenSinceYear',
                                               'Promo2SinceWeek',
                                               'Promo2SinceYear',
                                               'WeekOfYear'], axis=1)
        return data_store_X


if __name__ == '__main__':
    train = pd.read_csv('train.csv', low_memory=False, index_col='Date')
    test = pd.read_csv('test.csv', low_memory=False, index_col='Date')
    store = pd.read_csv('store.csv', low_memory=False)

    store_cleaned = clean_store(store)
    # print(store_cleaned.info())
    # print(store_cleaned.columns)

    X, y = clean_merge_data(train, store_cleaned, True)
    # print(X.info())
    # print(X.columns)
    # print(y)
    # X.to_csv('cleaned_train.csv')
    real_test = clean_merge_data(test,store_cleaned,False)
    print(real_test.info())
    print(real_test.columns)
    #
    # index = real_test['Open'] == 0
    # print(real_test.loc[index,'Open'])


    # real_test['new'] = [0]*len(real_test['Open'])
    # print(real_test)
    # X.to_csv('train_c.csv',index=False)
    # y.to_csv('test_c.csv',index=False)
