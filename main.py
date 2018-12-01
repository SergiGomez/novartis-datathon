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

## Training parameters
model_name = 'lgb'

if print_to_log == True:
    todayDate = dt.datetime.today().strftime('%Y%m%d')
    orig_stdout = sys.stdout
    f = open('main_' + todayDate+ '_'+ user_name_version +'.log', 'w')
    sys.stdout = f

t0 = time.time()
print("\n-- Raw data loading  --")
train = pd.read_csv(path_data + 'train_1.csv')
#test = pd.read_csv(path_data + 'key_1.csv')
start_time = time.time()
print("-- Data loading done: %s sec --" % np.round(time.time() - t0,1))

print('\tTraining set shape: {} Rows, {} Columns'.format(*train.shape))

#### Delete that for the competition !!!! ####
train = train.iloc[0:100,]
####

# Pre-processing
train_X = preprocessing.data_preprocessing(df = train)
print('\tShape train after Pre-processing: ', train_X.shape)
filename_preprocessed_data = os.path.join(path_data + 'train_preprocessed_' + user_name_version + '.csv')
train_X.to_csv(filename_preprocessed_data)

# Training
model = models.train_model(X = train_X, model_name = model_name)

# Prediction
test_Y = prediction.make_prediction(model = model, X = test_X)

# Prepare submission
submission = prediction.prepare_submission(test_Y)
