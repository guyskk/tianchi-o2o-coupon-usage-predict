import pandas as pd
from loguru import logger as LOG


def select_label(df, date_begin, date_end):
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
    selection = (
        df['coupon_id'].notnull()
        & df['date_received'].notnull()
        & (df['date_received'] >= date_begin)
        & (df['date_received'] <= date_end)
    )
    df = df.loc[selection, columns]
    return df


def select_feature(df, date_begin, date_end):
    cond1 = cond2 = True
    if 'date_received' in df.columns:
        col = df['date_received']
        cond1 = col.isnull() | ((date_begin <= col) & (col <= date_end))
    if 'date' in df.columns:
        col = df['date']
        cond2 = col.isnull() | ((date_begin <= col) & (col <= date_end))
    return df[cond1 & cond2]


def main():
    df = pd.read_msgpack(f'data/z1_raw_offline.msgpack')
    df_label_1 = select_label(df, '2016-04-14', '2016-05-14')
    df_label_2 = select_label(df, '2016-05-15', '2016-06-15')
    LOG.info('label size: offline={}, label_1={}, label_2={}',
             len(df), len(df_label_1), len(df_label_2))
    df_label_1.to_msgpack(f'data/z2_split_label_1.msgpack')
    df_label_2.to_msgpack(f'data/z2_split_label_2.msgpack')

    for df_name in ['offline', 'online_click', 'online_coupon']:
        df = pd.read_msgpack(f'data/z1_raw_{df_name}.msgpack')
        df_feature_1 = select_feature(df, '2016-01-01', '2016-04-13')
        df_feature_2 = select_feature(df, '2016-02-01', '2016-05-14')
        df_feature_3 = select_feature(df, '2016-03-15', '2016-06-30')
        LOG.info('{} feature size: all={}, feature_1={}, feature_2={}, feature_3={}',
                 df_name, len(df), len(df_feature_1), len(df_feature_2), len(df_feature_3))
        df_feature_1.to_msgpack(f'data/z2_split_{df_name}_feature_1.msgpack')
        df_feature_2.to_msgpack(f'data/z2_split_{df_name}_feature_2.msgpack')
        df_feature_3.to_msgpack(f'data/z2_split_{df_name}_feature_3.msgpack')


if __name__ == '__main__':
    main()
