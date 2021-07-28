from sklearn.ensemble import BaggingRegressor, VotingRegressor, AdaBoostRegressor
from MyMetrics import rmspe_origin

def get_bagging_reg_models(best_ridge, best_lasso, best_dt, X_train_std, y_train):
    """
    using three models which are already trained well.
    return :3 bagging models fitted
    """
    bag_ridge = BaggingRegressor(best_ridge, n_estimators=50, verbose=2)
    print('fitting bagging of ridge...')
    bag_ridge.fit(X_train_std, y_train)

    bag_lasso = BaggingRegressor(best_lasso, n_estimators=30, verbose=2)
    print('fitting bagging of lasso...')
    bag_lasso.fit(X_train_std, y_train)

    bag_dt = BaggingRegressor(best_dt, n_estimators=30, verbose=2)
    print('fitting bagging of dt...')
    bag_dt.fit(X_train_std, y_train)

    return bag_ridge, bag_lasso, bag_dt


def get_voting_reg(best_ridge, best_lasso, best_dt,
                   X_train_std, y_train):
    """
    using three models which are already trained well.
    return : voting regression fitted
    """
    vote = VotingRegressor(estimators=[
        ('ridge', best_ridge),
        ('lasso', best_lasso),
        ('decision tree', best_dt)
    ])
    print('fitting voting regression...')
    vote.fit(X_train_std, y_train)
    return (vote,)


# adaboost
def get_adaboost_models(best_ridge, best_lasso, best_dt,
                        X_train_std, y_train):
    ada_ridge = AdaBoostRegressor(best_ridge,n_estimators=30)
    ada_lasso = AdaBoostRegressor(best_lasso,n_estimators=30)
    ada_dt = AdaBoostRegressor(best_dt,n_estimators=30)

    print('fitting adaboost of ridge...')
    ada_ridge.fit(X_train_std, y_train)
    print('fitting adaboost of lasso...')
    ada_lasso.fit(X_train_std, y_train)
    print('fitting adaboost of desicion tree...')
    ada_dt.fit(X_train_std, y_train)

    return ada_ridge, ada_lasso, ada_dt


def get_best_model(best_ridge, best_lasso, best_dt, best_rf, X_train_std, y_train, X_test_std, y_test):
    """
    get the best model among those 3 models
    return : best model from ridge, lasso, decision tree
    """
    models = [model
              for sub in [get_bagging_reg_models(best_ridge, best_lasso, best_dt, X_train_std, y_train),
                          get_voting_reg(best_ridge, best_lasso, best_dt, X_train_std, y_train),
                          get_adaboost_models(best_ridge, best_lasso, best_dt, X_train_std, y_train)]
              for model in sub]

    best_ridge.fit(X_train_std, y_train)
    models.append(best_ridge)

    best_lasso.fit(X_train_std, y_train)
    models.append(best_lasso)

    best_dt.fit(X_train_std, y_train)
    models.append(best_dt)

    models.append(best_rf)

    # the smaller, the better
    scores = [rmspe_origin(y_test, model.predict(X_test_std)) for model in models]
    # print('---------------')
    # for score in scores:
    #     print(score)
    # print('--------------')
    return models[scores.index(min(scores))]
