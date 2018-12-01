import os
import pandas as pd
import numpy as np
import datetime as dt
import sys
import time

import preprocessing
import models
import prediction

## Basic execution params
path_data = '/Users/sergigomezpalleja/Downloads/'
user_name_version = 'sergi_1'
print_to_log = True
do_preprocess = True
do_modelling = False
do_prediction = False
do_submission = False

## Training parameters
model_name = 'lgb'

catVarsDict = {'Country' : 'LabelEncoder',
               'Brand_Group': 'LabelEncoder',
               'Cluster': 'LabelEncoder'}

if print_to_log == True:
    todayDate = dt.datetime.today().strftime('%Y%m%d')
    orig_stdout = sys.stdout
    f = open('main_' + todayDate+ '_'+ user_name_version +'.log', 'w')
    sys.stdout = f

t0 = time.time()
print("\n-- Raw data loading  --")
train = pd.read_csv(path_data + 'raw_data_master.csv')
start_time = time.time()
print("-- Data loading done: %s sec --" % np.round(time.time() - t0,1))
print('\tTraining set shape: {} Rows, {} Columns'.format(*train.shape))

# Pre-processing
if do_preprocess:
    train_X = preprocessing.data_preprocessing(df = train, catVarsDict = catVarsDict)
    print('\tShape train after Pre-processing: ', train_X.shape)
    filename_preprocessed_data = os.path.join(path_data + 'train_preprocessed_' + user_name_version + '.csv')
    train_X.to_csv(filename_preprocessed_data, index=False)
    print("OK")

else:
    #train_X = pd.read_csv(os.path.join(path_data))
    #print('\tShape of file Pre-processed: ', train_X.shape)
    pass

# Training
if do_modelling:
    model = models.train_model(X = train_X, model_name = model_name)

# Prediction
if do_prediction:
    test_Y = prediction.make_prediction(model = model, X = test_X)

# Prepare submission
if do_submission:
    submission = prediction.prepare_submission(test_Y)
