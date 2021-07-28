from EnsembleModels import get_best_model
from DataCleaning import clean_merge_data, clean_store
from MyMetrics import rmspe_origin
from RidgeModel import get_best_ridge
from LassoModel import get_best_lasso
from DecisionTreeModel import get_best_DecisionTreeRegressor
from RamdomForestModel import get_best_RandomForestRegressor

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# read data
train = pd.read_csv('train.csv', low_memory=False, index_col='Date')
test = pd.read_csv('test.csv', low_memory=False, index_col='Date')
store = pd.read_csv('store.csv', low_memory=False)

# clean data and merge them together
store_cleaned = clean_store(store)
X, y = clean_merge_data(train, store_cleaned, True)

# delete those outliers
# X = X[y < 12000]
# y = y[y < 12000]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=233)
real_test = clean_merge_data(test, store_cleaned, False)

# normalize the data
std_scaler = StandardScaler()
std_scaler.fit(X_train)
X_train_std = std_scaler.transform(X_train)
X_test_std = std_scaler.transform(X_test)
real_test_std = std_scaler.transform(real_test)

# get best base models
print('====================================Ridge=================================')
best_ridge = get_best_ridge(X_train_std, y_train, X_test_std, y_test)
print('====================================LASSO=================================')
best_lasso = get_best_lasso(X_train_std, y_train, X_test_std, y_test)
print('====================================Decision TREE=================================')
best_dt = get_best_DecisionTreeRegressor(X_train_std, y_train, X_test_std, y_test)
print('====================================Random Forest=================================')
best_rf = get_best_RandomForestRegressor(X_train_std, y_train, X_test_std, y_test)

# get best models
print('====================================Train best model=================================')
best_model = get_best_model(best_ridge, best_lasso, best_dt, best_rf, X_train_std, y_train, X_test_std, y_test)

print('best model:', best_model)
print('RMSPE score of best model:', rmspe_origin(y_test, y_pred=best_model.predict(X_test_std)))

# predict the real test data and save it into submission.csv
real_pred = best_model.predict(real_test_std)
submission = pd.DataFrame({'Id': test['Id'].values})
submission['Sales'] = real_pred

# another thing need to do is assign all cases whose Open == 0 to 0
index = (test['Open'] == 0).values
submission.loc[index, 'Sales'] = 0

# save to submission.csv
submission.to_csv('submission.csv', sep=',', index=False)
