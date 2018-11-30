import numpy as np
import time

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

params = {

    'walk-forward-cv' : {
        "n_splits" : 3 # sets the number of folds for cross-validation
    }

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

    values = X['value'].values

    tscv = TimeSeriesSplit(n_splits = params['walk-forward-cv']['n_splits'])

    # iterating over folds, train model on each, forecast and calculate error
    for train, val in tscv.split(values):

        model = 'the model chosen'

        predictions = model.result[-len(test):]

        error = rmse(predictions, actual)

        errors.append(error)

    print("\t Mean of errors: " np.mean(np.array(errors)))
    
    print("-- Training done: %s sec --" % np.round(time.time() - t0,1))

    return model

def rmse(y_true, y_pred):
    return round(np.sqrt(mean_squared_error(y_true, y_pred)), 5)
