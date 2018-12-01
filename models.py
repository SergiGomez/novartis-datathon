import numpy as np
import pandas as pd
import time

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.preprocessing import LabelEncoder, StandardScaler

import lightgbm as lgb

def train_model(X, model_name, params):

    t0 = time.time()

    errorsTotal = [] # errors array
    errorsIndividual =[]

    values_dates = X.date.unique()

    tscv = TimeSeriesSplit(n_splits = params['walk-forward-cv']['n_splits'])

    # iterating over folds, train model on each, forecast and calculate error
    for train, val in tscv.split(values_dates):

        print('Train: %s, Validation: %s' % (len(train), len(val)))

        dates_train = values_dates[train]
        dates_val = values_dates[val]

        train_X = X.loc[X.date.isin(dates_train)]
        val_X = X.loc[X.date.isin(dates_val)]

        # Date variable is not needed for training
        del train_X['date']
        del val_X['date']

        # Select those variables that will be used for training
        train_X = train_X[params['vars_model']]
        val_X = val_X[params['vars_model']]

        y_train = train_X.sales2.values
        y_val = val_X.sales2.values

        print("\t-- Start Fitting the model --")

        if model_name == 'lgb':
            model_trained = train_lgb(train_X = train_X,
                                      val_X = val_X,
                                      train_y = y_train,
                                      val_y = y_val,
                                      params = params)

            y_pred = model_trained.predict(val_X, num_iteration = model_trained.best_iteration)

        elif model_name == 'rf':
            # Number of trees in random forest
            n_estimators = [int(x) for x in np.linspace(20, 200, 10)]
            # Number of features to consider at every split
            max_features = ['auto', 'sqrt']
            # Maximum number of levels in tree
            max_depth = [int(x) for x in np.linspace(10, 100, 10)]
            # max_depth.append(None)
            # Minimum number of samples required to split a node
            min_samples_split = [2, 5, 10]
            # Minimum number of samples required at each leaf node
            min_samples_leaf = [1, 2, 4]
            # Method of selecting samples for training each tree
            bootstrap = [True, False]

            random_grid_rf = {'n_estimators': n_estimators,
                              'max_features': max_features,
                              'max_depth': max_depth,
                              'min_samples_split': min_samples_split,
                              'min_samples_leaf': min_samples_leaf,
                              'bootstrap': bootstrap}

            classifier_rf = RandomForestRegressor()
            rf_random = RandomizedSearchCV(estimator = classifier_rf,
                                            param_distributions = random_grid_rf,
                                            n_iter = 100,
                                            cv = 3,
                                            verbose = 1,
                                            random_state = 0,
                                            n_jobs = -1)

            model_trained = rf_random.fit(train_X, y_train)
            y_pred = model_trained.predict(val_X)

        elif model_name == 'lasso':
            scaler = StandardScaler()
            model = LassoCV()

            del train_X['sales2']
            del val_X['sales2']

            model_trained = model.fit(train_X, y_train)
            y_pred = model_trained.predict(val_X)

        all_val = val_X.copy()
        all_val['sales2'] = y_val
        all_val['y_hat'] = y_pred
        all_val.to_csv('/Users/sergigomezpalleja/Downloads/all_val.csv', index = False)

        totalAPE, individualAPE = prediction_absolute_prop_error(df = all_val)
        print('\t\t individualAPE: %s, totalAPE: %s' % (individualAPE, totalAPE))

        errorsTotal.append(totalAPE)
        errorsIndividual.append(individualAPE)

    print("\t\t Mean of Total APE: ", np.mean(np.array(errorsTotal)))
    print("\t\t Mean of Individual APE: ", np.mean(np.array(errorsIndividual)))

    print("-- Training done: %s sec --" % np.round(time.time() - t0,1))

    return model_trained

def rmse(y_true, y_pred):
    return round(np.sqrt(mean_squared_error(y_true, y_pred)), 5)

def train_lgb(train_X, val_X, train_y, val_y, params):

    lgb_train = lgb.Dataset(train_X, label = train_y)
    lgb_val = lgb.Dataset(val_X, label = val_y)

    model_trained = lgb.train(params = params['lgb'],
                      train_set = lgb_train,
                      num_boost_round = 500,
                      valid_sets = [lgb_train, lgb_val],
                      early_stopping_rounds = 100,
                      verbose_eval = 50)

    return model_trained

def prediction_absolute_prop_error(df):

    y_true = df.sales2.sum()
    y_pred = df.y_hat.sum()
    totalAPE = np.abs(y_true - y_pred)*100 / y_true

    df_groups = prepare_dataframe_groups(df)

    df_groups['sales2_denominator'] = df_groups['sales2']
    df_groups.loc[df_groups.sales2_denominator == 0.0, 'sales2_denominator']= 1.0

    df_groups['error'] = np.abs(df_groups['y_hat'] - df_groups['sales2'])*100 / df_groups['sales2_denominator']

    individualAPE = df_groups['error'].mean()

    return totalAPE, individualAPE

def prepare_dataframe_groups(df):

    df_group1 = df.loc[df['Brand_Group'].isin([51, 73, 90])]
    df_group2 = df.loc[df['Brand_Group'].isin([96,97])]
    df_group3 = df.loc[~df['Brand_Group'].isin([17, 24, 30, 31, 36, 41, 51, 73, 90, 96, 97])]
    df_group_all = df.loc[df['Brand_Group'].isin([17, 24, 30, 31, 36, 41])]

    df_group1['Brand_Group'] = 'Brand Group 51, 73, 90'
    df_group2['Brand_Group'] = 'Brand Group 96, 97'
    df_group3['Brand_Group'] = 'others'
    df_group_all['Brand_Group'] = 'Brand Group ' + df_group_all['Brand_Group'].astype(str)

    df_group1_sum = df_group1.groupby(['Cluster', 'month', 'year', 'Brand_Group']).sum().reset_index()
    df_group2_sum = df_group2.groupby(['Cluster', 'month', 'year', 'Brand_Group']).sum().reset_index()
    df_group3_sum = df_group3.groupby(['Cluster', 'month', 'year', 'Brand_Group']).sum().reset_index()
    df_group_all_sum = df_group_all.groupby(['Cluster', 'month', 'year', 'Brand_Group']).sum().reset_index()

    vars_to_keep = ['Cluster', 'month', 'year', 'Brand_Group', 'sales2', 'y_hat']
    df_group1_sum = df_group1_sum[vars_to_keep]
    df_group2_sum = df_group2_sum[vars_to_keep]
    df_group3_sum = df_group3_sum[vars_to_keep]
    df_group_all_sum = df_group_all_sum[vars_to_keep]

    df_all = pd.concat([df_group1_sum, df_group2_sum, df_group3_sum, df_group_all_sum])

    df_all['Cluster'] = 'Cluster ' + df_all['Cluster'].astype(str)

    return df_all
