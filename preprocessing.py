import pandas as pd
import numpy as np
import time

def data_preprocessing(df):

    print('\n-- Data Pre-processing --')
    t0 = time.time()

    # If data is in Wide format, we need to flatten it
    df = pd.melt(df[list(df.columns[-50:])+['Page']],
                        id_vars='Page', var_name='date', value_name='Visits')

    # Convert date to datetime format
    df['date'] = df['date'].astype('datetime64[ns]')

    # Identify if it's weekend
    df['weekend'] = ((df.date.dt.dayofweek) // 5 == 1).astype(float)

    print("-- Pre-processing done: %s sec --" % np.round(time.time() - t0,1))

    return df
