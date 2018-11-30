import os
import pandas as pd
import numpy as np
import datetime as dt
import sys
import time

import preprocessing

## Basic execution params
path_data = '/Users/sergigomezpalleja/Downloads/'
user_name_version = 'sergi_1'
print_to_log = True

if print_to_log == True:
    todayDate = dt.datetime.today().strftime('%Y%m%d')
    orig_stdout = sys.stdout
    f = open('main_' + todayDate+ '_'+ user_name +'.log', 'w')
    sys.stdout = f

t0 = time.time()
print("\n-- Data loading  --")
train = pd.read_csv(path_data + 'train_1.csv')
#test = pd.read_csv(path_data + 'key_1.csv')
start_time = time.time()
print("-- Data loading done: %s sec --" % np.round(time.time() - t0,1))

print('\tTraining set shape: {} Rows, {} Columns'.format(*train.shape))

#### Delete that for the competition !!!! ####
train = train.iloc[0:100,]
####

train_X = preprocessing.data_preprocessing(df = train)
print('\tShape train after Pre-processing: ', train_X.shape)
filename_preprocessed_data = os.path.join(path_data + 'train_preprocessed_' + user_name + '.csv')
train_X.to_csv(filename_preprocessed_data)
