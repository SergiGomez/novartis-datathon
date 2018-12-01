import pandas as pd
import numpy as np
import time

def data_preprocessing(df):

    print('\n-- Data Pre-processing --')
    t0 = time.time()

    # If data is in Wide format, we need to flatten it
    key_vars= ['Cluster',
                'Brand Group',
                'Country',
                'Function']

    df = pd.melt(df[list(df.columns[-50:])+ key_vars],
                        id_vars= key_vars, var_name='date', value_name='sales')

    # Convert date to datetime format
    #df['date'] = df['date'].astype('datetime64[ns]')

    # Identify if it's weekend
    #df['weekend'] = ((df.date.dt.dayofweek) // 5 == 1).astype(float)

    print("-- Pre-processing done: %s sec --" % np.round(time.time() - t0,1))

    return df

def stationarize_series(df):

    ads_diff = ads.Ads - ads.Ads.shift(24)
