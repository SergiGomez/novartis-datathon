import numpy as np
import time

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

import lightgbm as lgb

params = {

    'walk-forward-cv' : {
        "n_splits" : 5 # sets the number of folds for cross-validation
    },

    'lgb' : {
        "objective" : "regression",
        "metric" : "rmse",
        "num_leaves" : 40,
        "learning_rate" : 0.005,
        "bagging_fraction" : 0.6,
        "feature_fraction" : 0.6,
        "bagging_frequency" : 6,
        "bagging_seed" : 42,
        "verbosity" : -1,
        "seed": 42}

}

def train_model(X, model_name, params = params):

    t0 = time.time()

    print("\t-- Start Fitting the model --")

    errors = [] # errors array

    values_dates = X.date.unique()

    tscv = TimeSeriesSplit(n_splits = params['walk-forward-cv']['n_splits'])

    # iterating over folds, train model on each, forecast and calculate error
    for train, val in tscv.split(values_dates):

        print('Train: %s, Validation: %s' % (len(train), len(val)))

        dates_train = values_dates[train]
        dates_val = values_dates[val]

#        if model_name == 'lgb':

    #        model = train_lgb(train_X = train_X, val_X = val_X, params = params)

    #    predictions = model.result[-len(val):]

    #    error = rmse(predictions, actual)

    #    errors.append(error)

    print("\t Mean of errors: ", np.mean(np.array(errors)))

    print("-- Training done: %s sec --" % np.round(time.time() - t0,1))

    #return model
    return 10

def rmse(y_true, y_pred):
    return round(np.sqrt(mean_squared_error(y_true, y_pred)), 5)

def train_lgb(train_X, val_X, params):

    train_y = train_X['ventas']
    val_y = val_y['ventas']

    del train_X['ventas']
    del val_X['ventas']

    lgb_train = lgb.Dataset(train_X, label = train_y)
    lgb_val = lgb.Dataset(val_X, label = val_y)

    model = lgb.train(params = params['lgb'],
                      lgb_train = lgb_train,
                      num_boost_round = 500,
                      valid_sets = [lgb_train, lgb_val],
                      early_stopping_rounds = 100,
                      verbose_eval = 50)

    train_y_pred = model.predict(train_X, num_iteration = model.best_iteration)
    val_y_pred = model.predict(val_X, num_iteration = model.best_iteration)
    print(f"LGBM: RMSE val: {rmse(val_y, val_y_pred)}  - RMSE train: {rmse(train_y, train_y_pred)}")

    return model
