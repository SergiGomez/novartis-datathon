import numpy as np
import time

from sklearn.model_selection import TimeSeriesSplit

params = {

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
    
    print("-- Training done: %s sec --" % np.round(time.time() - t0,1))

    return model
