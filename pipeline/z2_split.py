import os
import pandas as pd
import numpy as np
from loguru import logger as LOG

DATE_SPLIT = pd.to_datetime('2016-04-01')


def get_split_type():
    return os.getenv('SPLIT_TYPE', 'date')


def split_label(df):
    columns = [
        'user_id',
        'merchant_id',
        'coupon_id',
        'distance',
        'date_received',
        'discount_name',
        'is_dazhe',
        'is_manjian',
        'discount_man',
        'discount_jian',
        'discount_rate',
        'label',
    ]
    selection = (df['coupon_id'].notnull()
                 & df['date_received'].notnull()
                 & (df['date_received'] >= DATE_SPLIT))
    df = df.loc[selection, columns]
    if get_split_type() == 'date':
        LOG.info('split label by date')
        df_train = df[df['date_received'] < '2016-06-01']
        df_validate = df[df['date_received'] >= '2016-06-01']
    else:
        LOG.info('split label by random')
        mask = np.random.rand(len(df)) < 0.9
        df_train = df[mask]
        df_validate = df[~mask]
    return df_train, df_validate


def select_data(df):
    cond1 = cond2 = None
    if 'date_received' in df.columns:
        cond1 = df['date_received'].isnull() | (df['date_received'] < DATE_SPLIT)
    if 'date' in df.columns:
        cond2 = df['date'].isnull() | (df['date'] < DATE_SPLIT)
    if cond1 is None:
        if cond2 is None:
            return df
        else:
            return df[cond2]
    else:
        if cond2 is None:
            return df[cond1]
        else:
            return df[cond1 & cond2]


def main():
    df = pd.read_msgpack(f'data/z1_raw_offline.msgpack')
    df_train, df_validate = split_label(df)
    LOG.info('dataset size: offline={}, train={}, validate={}',
             len(df), len(df_train), len(df_validate))
    df_train.to_msgpack(f'data/z2_split_train.msgpack')
    df_validate.to_msgpack(f'data/z2_split_validate.msgpack')

    df = select_data(df)
    df.to_msgpack('data/z2_split_offline.msgpack')

    df = pd.read_msgpack(f'data/z1_raw_online_click.msgpack')
    df = select_data(df)
    df.to_msgpack(f'data/z2_split_online_click.msgpack')

    df = pd.read_msgpack(f'data/z1_raw_online_coupon.msgpack')
    df = select_data(df)
    df.to_msgpack(f'data/z2_split_online_coupon.msgpack')


if __name__ == '__main__':
    main()
