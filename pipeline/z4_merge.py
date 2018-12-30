import numpy as np
import pandas as pd
from z3_feature import NULL_DISTANCE
from loguru import logger as LOG


def read_features(num):
    feature_names = ['user', 'merchant', 'coupon', 'discount', 'distance']
    features = {}
    for name in feature_names:
        path = 'data/z3_feature_{}_{}.msgpack'.format(name, num)
        LOG.info('#{} read {} -> {}', num, name, path)
        df = pd.read_msgpack(path)
        features[name] = df
    return features


def read_label(num):
    path = 'data/z3_feature_label_{}.msgpack'.format(num)
    LOG.info('#{} read label -> {}', num, path)
    return pd.read_msgpack(path)


def merge_feratures(features, label):
    df = label

    df_user = features['user']
    df = pd.merge(df, df_user, left_on='user_id', right_index=True, how='left')

    df_merchant = features['merchant']
    df = pd.merge(df, df_merchant, left_on='merchant_id', right_index=True, how='left')

    df_coupon = features['coupon']
    df = pd.merge(df, df_coupon, left_on='coupon_id', right_index=True, how='left')

    df_discount = features['discount']
    df = pd.merge(df, df_discount, left_on='discount_name', right_index=True, how='left')

    df_distance = features['distance']
    df = pd.merge(df, df_distance, left_on='distance', right_index=True, how='left')
    df['distance'].fillna(NULL_DISTANCE, inplace=True)
    df_full = df.copy()

    df = df.drop([
        'user_id', 'merchant_id', 'coupon_id', 'discount_name',
        'date_received', 'date_received_name', 'date', 'date_name',
    ], axis=1, errors='ignore')

    df.fillna(0, inplace=True)
    bool_columns = df.dtypes[df.dtypes == np.dtype(object)].index
    df[bool_columns] = df[bool_columns].astype(bool)
    int_columns = df.dtypes[df.dtypes == np.dtype(int)].index
    df[int_columns] = df[int_columns].astype(float)

    return df, df_full


def main():
    for num in [1, 2, 3]:
        LOG.info('merge features for dataset {}', num)
        features = read_features(num)
        label = read_label(num)
        df, df_full = merge_feratures(features, label)
        LOG.info('dataset {}: size={}, columns={}', num, len(df), len(df.columns))
        df.to_msgpack('data/z4_merge_{}.msgpack'.format(num))
        df_full.to_msgpack('data/z4_merge_{}_full.msgpack'.format(num))


if __name__ == "__main__":
    main()
