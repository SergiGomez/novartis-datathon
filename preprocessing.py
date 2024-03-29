import pandas as pd
import numpy as np
import time

from sklearn.preprocessing import StandardScaler

import category_encoders

def data_preprocessing(df, catVarsDict):

    print('\n-- Data Pre-processing --')
    t0 = time.time()

    df = char_to_int(df, catVarsDict)

    df = pd.melt(df[list(df.columns[-50:])+ key_vars],
                        id_vars= key_vars, var_name='date', value_name='sales')

    # Convert date to datetime format
    df['date'] = df['date'].astype('datetime64[ns]')

    # Add month and year
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month

    df = treatment_missings(df)

    df = treatment_target(df, var = 'sales2')

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
        elif (value == 'OneHot'):
            encoder = category_encoders.one_hot.OneHotEncoder(cols = [var],
                                            drop_invariant = True,
                                            return_df = True,
                                            use_cat_names = True)
            df = encoder.fit_transform(df)

    return df

def treatment_missings(df):

    print("\t Percentage Missings Inv 1 ", df.loc[df.inv1.isnull()].shape[0]/df.shape[0])
    print("\t Percentage Missings Inv 2 ", df.loc[df.inv2.isnull()].shape[0]/df.shape[0])
    print("\t Percentage Missings Inv 3 ", df.loc[df.inv3.isnull()].shape[0]/df.shape[0])
    print("\t Percentage Missings Inv 4 ", df.loc[df.inv4.isnull()].shape[0]/df.shape[0])
    print("\t Percentage Missings Inv 5 ", df.loc[df.inv5.isnull()].shape[0]/df.shape[0])
    print("\t Percentage Missings Inv 6 ", df.loc[df.inv6.isnull()].shape[0]/df.shape[0])

    # Investments missing to 0
    df['inv1'].fillna(value = 0.0, inplace = True)
    df['inv2'].fillna(value = 0.0, inplace = True)
    df['inv3'].fillna(value = 0.0, inplace = True)
    df['inv4'].fillna(value = 0.0, inplace = True)
    df['inv5'].fillna(value = 0.0, inplace = True)
    df['inv6'].fillna(value = 0.0, inplace = True)

    return df

def treatment_zeros_target(df):

    # Sales 2 missings observations are deleted
    df = df.loc[~df.sales2.isnull()]

    df = df.loc[df.sales2 != 0.0]

    df = df.loc[df.sales2 > 0.0]

    return df

def treatment_target(df, var):

    df[var] = np.log(df[var])

    return df

def inverse_treatment_target(df, var):

    df[var] = np.exp(df[var])

    return df
