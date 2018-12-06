import os
import pandas as pd
import numpy as np
import datetime as dt
import sys
import time

import preprocessing
import models
import prediction
import ini_params

import warnings
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

## Basic execution params
path_data = '/Users/sergigomezpalleja/Downloads/'
user_name_version = 'sergi_6'
print_to_log = False
add_feats = True
do_preprocess = True
do_modelling = True
do_prediction = True
do_submission = False

model_name = 'rf'

catVarsDict = ini_params.catVarsDict
feats_to_add  = ini_params.feats_to_add
params = ini_params.params
feats_join = ini_params.feats_join
feats_to_add = ini_params.feats_to_add

if print_to_log == True:
    todayDate = dt.datetime.today().strftime('%Y%m%d')
    orig_stdout = sys.stdout
    f = open('main_' + todayDate+ '_'+ user_name_version +'.log', 'w')
    sys.stdout = f

t0 = time.time()
print("\n-- Raw data loading  --")
df = pd.read_csv(path_data + 'raw_data_master.csv')
start_time = time.time()
print("-- Data loading done: %s sec --" % np.round(time.time() - t0,1))
print('\tTraining set shape: {} Rows, {} Columns'.format(*df.shape))

# Adding feats extracted by the tream
if add_feats:
    df_feats = pd.read_csv(path_data + 'preprocessed_data_master.csv')
    df_feats = df_feats[ feats_join + feats_to_add]
    df_feats_scale = df_feats[ feats_to_add]
    df_feats_no_scale = df_feats[feats_join]
    scaler = StandardScaler()
    df_feats_scale = scaler.fit(df_feats_scale)
    df_feats = pd.concat([df_feats_no_scale, df_feats_scale])
    df = pd.merge(df, df_feats, on = feats_join, how = 'outer')


# Pre-processing
if do_preprocess:
    df_processed = preprocessing.data_preprocessing(df = df, catVarsDict = catVarsDict)
    print('\tShape train after Pre-processing: ', df_processed.shape)
    filename_preprocessed_data = os.path.join(path_data + 'train_preprocessed_' + user_name_version + '.csv')
    df_processed.to_csv(filename_preprocessed_data, index=False)
else:
    filename_preprocessed_data = os.path.join(path_data + 'train_preprocessed_R_1.csv')
    df_processed = pd.read_csv(filename_preprocessed_data)
    df_processed['date'] = df_processed['date'].astype('datetime64[ns]')
    #df_processed = preprocessing.treatment_target(df_processed, var = 'sales2')

# Training
# Keeping the test set aside
train_X = df_processed.loc[df_processed.date < '2018-01-01']
test_X = df_processed.loc[df_processed.date >= '2018-01-01']

if do_modelling:
    model_trained = models.train_model(X = train_X, model_name = model_name, params = params)

# Prediction
if do_prediction:
    test_X = test_X[params['vars_model']]
    test_X.to_csv('/Users/sergigomezpalleja/Downloads/test_X.csv', index = False)
    # Select those variables that will be used for training
    if model_name == 'lgb':
        test_Y = model_trained.predict(test_X, num_iteration = model_trained.best_iteration)
    elif model_name == 'rf':
        test_Y = model_trained.predict(test_X)
    test_all = test_X.copy()
    test_all['y_hat'] = test_Y
    test_all = preprocessing.inverse_treatment_target(df = test_all, var = 'y_hat')
    filename_test_predictions = os.path.join(path_data + 'test_predictions_' + user_name_version + '.csv')
    test_all.to_csv(filename_test_predictions, index=False)

# Prepare submission
if do_submission:
    submission = prediction.prepare_submission(test_Y)
