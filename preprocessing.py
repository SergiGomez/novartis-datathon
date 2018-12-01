import pandas as pd
import numpy as np
import time

def data_preprocessing(df, catVarsDict):

    print('\n-- Data Pre-processing --')
    t0 = time.time()

    df = char_to_int(df, catVarsDict)

    #df = pd.melt(df[list(df.columns[-50:])+ key_vars],
    #                    id_vars= key_vars, var_name='date', value_name='sales')

    # Convert date to datetime format
    df['date'] = df['date'].astype('datetime64[ns]')

    # Identify if it's weekend
    #df['weekend'] = ((df.date.dt.dayofweek) // 5 == 1).astype(float)

    print("-- Pre-processing done: %s sec --" % np.round(time.time() - t0,1))

    return df

def stationarize_series(df):

    ads_diff = ads.Ads - ads.Ads.shift(24)

def char_to_int(df, catVarsDict):

    for var , value in catVarsDict.items():

        if (value == 'BinaryEncoder'):
            encoder = category_encoders.BinaryEncoder(cols = [var],
                                            drop_invariant = True,
                                            return_df = True)
            df = encoder.fit_transform(df)
        elif (value == 'LabelEncoder'):
            df[var] = df[var].str.extract('(\d+)').astype(int)
            #df[var] = LabelEncoder().fit_transform(df[var])
        elif (value == 'OneHot'):
            encoder = category_encoders.one_hot.OneHotEncoder(cols = [var],
                                            drop_invariant = True,
                                            return_df = True,
                                            use_cat_names = True)
            df = encoder.fit_transform(df)

    return df
