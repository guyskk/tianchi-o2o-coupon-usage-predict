import pandas as pd
import numpy as np
from collections import namedtuple
from loguru import logger as LOG
from pandas_parallel import pandas_parallel, cloudpickle_safe_lru_cache


FeatureDataset = namedtuple(
    'FeatureDataset', 'offline online_click online_coupon')


def most_freq(s):
    c = s.mode()
    return c[0] if len(c) > 0 else np.nan


def flat_columns(df, format=None):
    if format:
        columns = [format.format(*x) for x in df.columns.ravel()]
    else:
        columns = ['_'.join(map(str, x)) for x in df.columns.ravel()]
    df.columns = columns


def prefix_columns(df, prefix):
    if not prefix.endswith('_'):
        prefix = prefix + '_'
    df.columns = [prefix + x for x in df.columns]


NULL_DISTANCE = 12

HOLIDAY_2016 = {
    '01-01',  # 元旦
    '01-02',
    '01-03',
    '02-01',  # 小年
    '02-04',  # 立春
    '02-07',
    '02-08',  # 春节
    '02-09',
    '02-10',
    '02-11',
    '02-12',
    '02-13',
    '02-14',  # 情人节
    '02-22',  # 元宵节
    '03-08',  # 妇女节
    '03-12',  # 植树节
    '03-15',  # 消费者权益日
    '04-01',  # 愚人节
    '04-02',
    '04-03',
    '04-04',  # 清明
    '04-30',
    '05-01',  # 劳动节
    '05-02',
    '05-08',  # 母亲节
    '06-01',  # 儿童节
    '06-07',  # 高考
    '06-08',
    '06-09',  # 端午
    '06-10',
    '06-11',
    '06-19',  # 父亲节
}
HOLIDAY_2016 = set([tuple(map(int, x.split('-'))) for x in HOLIDAY_2016])
JIABAN_2016 = {
    '02-06',
    '02-14',
    '06-12',
}
JIABAN_2016 = set([tuple(map(int, x.split('-'))) for x in JIABAN_2016])


@cloudpickle_safe_lru_cache(maxsize=365)
def date_to_vec(x):
    if pd.isnull(x):
        return [np.nan for _ in range(len(DATE_VEC_COLUMNS))]
    date = (x.month, x.day)
    is_holiday = date in HOLIDAY_2016
    is_jiaban = date in JIABAN_2016
    vec = [
        is_holiday,
        is_jiaban,
        x.is_month_start,
        x.is_month_end,
    ]
    vec.extend([i == x.dayofweek for i in range(7)])
    name = []
    if is_holiday:
        name.append('吉')
    if is_jiaban:
        name.append('班')
    if x.is_month_start:
        name.append('月初')
    if x.is_month_end:
        name.append('月末')
    name.append('周' + str(x.dayofweek + 1))
    name = ''.join(name)
    vec.insert(0, name)
    return vec


DATE_VEC_COLUMNS = [
    'name',
    'is_holiday',
    'is_jiaban',
    'is_month_start',
    'is_month_end',
    'is_weekday_1',
    'is_weekday_2',
    'is_weekday_3',
    'is_weekday_4',
    'is_weekday_5',
    'is_weekday_6',
    'is_weekday_7',
]


def apply_date_to_vec(df, column):
    s = df[column].apply(date_to_vec)
    df_vec = pd.DataFrame.from_records(
        s.tolist(), index=s.index, columns=DATE_VEC_COLUMNS)
    prefix_columns(df_vec, column)
    return df_vec


def compute_user_feature_is_active_by(df, date_column):
    # !!! Do NOT parallel
    df_last_date = df.groupby('user_id')[date_column].max()\
        .to_frame('last_date')\
        .sort_values('last_date', ascending=False)
    df_is_active = df_last_date.index.to_series()
    df_is_active[:] = False
    df_is_active = df_is_active.to_frame('is_active')
    active_user_idx = df_last_date.iloc[:int(len(df_last_date)*0.2)].index
    df_is_active.loc[active_user_idx] = True
    return df_is_active


def compute_user_feature_is_active(df, df_name):
    feature_dfs = []
    for col in ['date', 'date_received']:
        if col in df.columns:
            df_is_active = compute_user_feature_is_active_by(df, col)
            prefix_columns(df_is_active, f'{df_name}_{col}')
            feature_dfs.append(df_is_active)
    return pd.concat(feature_dfs, axis=1)


def compute_user_feature_merchant(df):
    g = df.groupby('user_id')['merchant_id']
    df_merchant_nunique = g.nunique().to_frame('merchant_nunique')
    df_merchant_count = g.size().to_frame('merchant_count')
    df_merchant_value_stats = df.groupby('user_id')['merchant_id']\
        .value_counts()\
        .groupby('user_id')\
        .agg(['max', 'mean', 'median', most_freq])
    prefix_columns(df_merchant_value_stats, 'merchant_value_count')
    return pd.concat([
        df_merchant_nunique,
        df_merchant_count,
        df_merchant_value_stats,
    ], axis=1)


def compute_feature_to_user(df, by):
    g = df.groupby(by)['user_id']
    df_user_nunique = g.nunique().to_frame('user_nunique')
    df_user_count = g.size().to_frame('user_count')
    df_usage_stats = df.groupby(by)['user_id']\
        .value_counts()\
        .groupby(by)\
        .agg(['min', 'max', 'median', 'mean', most_freq])
    prefix_columns(df_usage_stats, 'user_usage')
    return pd.concat([
        df_user_nunique,
        df_user_count,
        df_usage_stats,
    ], axis=1)


def compute_merchant_feature_to_user(df):
    return compute_feature_to_user(df, by='merchant_id')


def compute_coupon_feature_to_user(df):
    return compute_feature_to_user(df, by='coupon_id')


def compute_discount_feature_to_user(df):
    return compute_feature_to_user(df, by='discount_name')


def compute_distance_feature_to_user(df):
    return compute_feature_to_user(df, by='distance')


def compute_feature_discount(df, by):
    # LOG.info('coupon counts')
    g = df.groupby(by)['coupon_id']
    df_coupon_count = g.size().to_frame(f'coupon_count')
    df_coupon_nunique = g.nunique().to_frame(f'coupon_nunique')
    df_coupon_value_stats = df.groupby(by)['coupon_id']\
        .value_counts()\
        .groupby(by)\
        .agg(['max', 'mean', 'median', most_freq])
    prefix_columns(df_coupon_value_stats, 'coupon_value_count')
    # LOG.info('discount stats')
    df_discount_stats = df.groupby(by)\
        .agg({
            'discount_man': ['max', 'min', 'median', 'mean', most_freq],
            'discount_jian': ['max', 'min', 'median', 'mean', most_freq],
            'discount_rate': ['max', 'min', 'median', 'mean', most_freq],
        })
    flat_columns(df_discount_stats)
    # LOG.info('discount type counts')
    discount_types = ['xianshi', 'manjian', 'dazhe']
    df_discount_type_counts = []
    for discount_type in discount_types:
        if discount_type in df.columns:
            df_sub = df[df[f'is_{discount_type}']]\
                .groupby(by)\
                .size().to_frame(f'{discount_type}_count')
            df_discount_type_counts.append(df_sub)
    return pd.concat([
        df_coupon_count,
        df_coupon_nunique,
        df_coupon_value_stats,
        df_discount_stats,
        *df_discount_type_counts,
    ], axis=1)


def compute_user_feature_coupon(df):
    return compute_feature_discount(df, by='user_id')


def compute_merchant_feature_coupon(df):
    return compute_feature_discount(df, by='merchant_id')


def compute_feature_distance(df, by):
    df_feature = df.groupby(by)['distance']\
        .agg(['max', 'min', 'median', 'mean', most_freq])
    prefix_columns(df_feature, 'distance')
    return df_feature


def compute_user_feature_distance(df):
    return compute_feature_distance(df, by='user_id')


def compute_merchant_feature_distance(df):
    return compute_feature_distance(df, by='merchant_id')


def compute_feature_date(df, date_column, by):
    df_date_vec = apply_date_to_vec(df, date_column)
    df = pd.concat([df, df_date_vec], axis=1)
    cols = df_date_vec.columns[1:]
    df_date_vec_stats = df.groupby(by)[cols].any()
    return df_date_vec_stats


def compute_user_feature_date(df, date_column):
    return compute_feature_date(df, date_column, by='user_id')


def compute_merchant_feature_date(df, date_column):
    return compute_feature_date(df, date_column, by='merchant_id')


def compute_feature_discount_hold_days(df, by):
    df_coupon_hold_days = (df['date'] - df['date_received'])\
        .apply(lambda x: x.days)\
        .to_frame('coupon_hold_days')
    df = pd.concat([df, df_coupon_hold_days], axis=1)
    df_feature = df.groupby(by)['coupon_hold_days']\
        .agg(['max', 'min', 'median', 'mean', most_freq])
    prefix_columns(df_feature, 'coupon_hold_days')
    return df_feature


def compute_user_feature_coupon_hold_days(df):
    return compute_feature_discount_hold_days(df, by='user_id')


def compute_transfer_rate(df, from_column, to_column):
    name = 'transfer_rate_' + from_column + '_to_' + to_column
    df = (df[to_column] / df[from_column]).to_frame(name)
    return df


def compute_merchant_feature_coupon_hold_days(df):
    return compute_feature_discount_hold_days(df, by='merchant_id')


def compute_discount_feature(df, df_name):
    dfs = {
        'received_coupon': df[df.date.isnull() & df.coupon_id.notnull()],
        'buy_with_coupon': df[df.date.notnull() & df.coupon_id.notnull()],
    }
    feature_dfs = []
    for name, df_sub in dfs.items():
        df_sub = compute_discount_feature_to_user(df_sub)
        prefix_columns(df_sub, name)
        feature_dfs.append(df_sub)
    df_result = pd.concat(feature_dfs, axis=1, sort=False)
    prefix_columns(df_result, df_name)
    return df_result


def compute_predict_feature(df):
    df = df[df['coupon_id'].notnull() & df['date_received'].notnull()]
    df_date_received_vec = apply_date_to_vec(df, 'date_received')
    df = pd.concat([df, df_date_received_vec], axis=1)
    return df


@pandas_parallel(partitioner='hash', partition_column='user_id', progress_bar=True)
def extract_user_feature_offline(df, df_name):
    dfs = {
        'received_coupon': df[df.date.isnull() & df.coupon_id.notnull()],
        'buy_with_coupon': df[df.date.notnull() & df.coupon_id.notnull()],
        'buy_without_coupon': df[df.date.notnull() & df.coupon_id.isnull()],
    }
    df_received_coupon = dfs['received_coupon']
    df_buy_with_coupon = dfs['buy_with_coupon']
    df_buy_without_coupon = dfs['buy_without_coupon']
    feature_dfs = [
        ('received_coupon', compute_user_feature_coupon(df_received_coupon)),
        ('buy_with_coupon', compute_user_feature_coupon(df_buy_with_coupon)),
        ('buy_with_coupon', compute_user_feature_coupon_hold_days(df_buy_with_coupon)),
        ('received_coupon', compute_user_feature_date(df_received_coupon, 'date_received')),
        ('buy_with_coupon', compute_user_feature_date(df_buy_with_coupon, 'date_received')),
        ('buy_with_coupon', compute_user_feature_date(df_buy_with_coupon, 'date')),
        ('buy_without_coupon', compute_user_feature_date(df_buy_without_coupon, 'date')),
        ('buy_with_coupon', compute_user_feature_distance(df_buy_with_coupon)),
        ('buy_without_coupon', compute_user_feature_distance(df_buy_without_coupon)),
        ('received_coupon', compute_user_feature_merchant(df_received_coupon)),
        ('buy_with_coupon', compute_user_feature_merchant(df_buy_with_coupon)),
        ('buy_without_coupon', compute_user_feature_merchant(df_buy_without_coupon)),
    ]
    results = []
    for prefix, df_feat in feature_dfs:
        prefix_columns(df_feat, prefix)
        results.append(df_feat)
    df_result = pd.concat(results, axis=1)
    df_transfer_rate = compute_transfer_rate(
        df_result, 'received_coupon_coupon_count', 'buy_with_coupon_coupon_count')
    df_result = pd.concat([df_result, df_transfer_rate], axis=1)
    prefix_columns(df_result, df_name)
    return df_result


@pandas_parallel(partitioner='hash', partition_column='user_id', progress_bar=True)
def extract_user_feature_online_click(df, df_name):
    df_click_count = df.groupby('user_id')['count'].agg(['max', 'median'])
    prefix_columns(df_click_count, 'click_count_')
    df_merchant_stats = compute_user_feature_merchant(df)
    df_date_vec_stats = compute_user_feature_date(df, 'date')
    df_result = pd.concat([
        df_click_count,
        df_merchant_stats,
        df_date_vec_stats,
    ], axis=1)
    prefix_columns(df_result, df_name)
    return df_result


@pandas_parallel(partitioner='hash', partition_column='user_id', progress_bar=True)
def extract_user_feature_online_coupon(df, df_name):
    dfs = {
        'received_coupon': df[df.date.isnull() & df.coupon_id.notnull()],
        'buy_with_coupon': df[df.date.notnull() & df.coupon_id.notnull()],
        'buy_without_coupon': df[df.date.notnull() & df.coupon_id.isnull()],
    }
    df_received_coupon = dfs['received_coupon']
    df_buy_with_coupon = dfs['buy_with_coupon']
    df_buy_without_coupon = dfs['buy_without_coupon']
    feature_dfs = [
        ('received_coupon', compute_user_feature_coupon(df_received_coupon)),
        ('buy_with_coupon', compute_user_feature_coupon(df_buy_with_coupon)),
        ('buy_with_coupon', compute_user_feature_coupon_hold_days(df_buy_with_coupon)),
        ('received_coupon', compute_user_feature_date(df_received_coupon, 'date_received')),
        ('buy_with_coupon', compute_user_feature_date(df_buy_with_coupon, 'date_received')),
        ('buy_with_coupon', compute_user_feature_date(df_buy_with_coupon, 'date')),
        ('buy_without_coupon', compute_user_feature_date(df_buy_without_coupon, 'date')),
        ('received_coupon', compute_user_feature_merchant(df_received_coupon)),
        ('buy_with_coupon', compute_user_feature_merchant(df_buy_with_coupon)),
        ('buy_without_coupon', compute_user_feature_merchant(df_buy_without_coupon)),
    ]
    results = []
    for prefix, df_feat in feature_dfs:
        prefix_columns(df_feat, prefix)
        results.append(df_feat)
    df_result = pd.concat(results, axis=1)
    df_transfer_rate = compute_transfer_rate(
        df_result, 'received_coupon_coupon_count', 'buy_with_coupon_coupon_count')
    df_result = pd.concat([df_result, df_transfer_rate], axis=1)
    prefix_columns(df_result, df_name)
    return df_result


def extract_user_feature(dataset):
    feature_dfs = []
    df = dataset.offline
    feature_dfs.append(extract_user_feature_offline(df, 'user_offline'))
    feature_dfs.append(compute_user_feature_is_active(df, 'user_offline'))
    df = dataset.online_click
    feature_dfs.append(extract_user_feature_online_click(df, 'user_online_click'))
    feature_dfs.append(compute_user_feature_is_active(df, 'user_online_click'))
    df = dataset.online_coupon
    feature_dfs.append(extract_user_feature_online_coupon(df, 'user_online_coupon'))
    feature_dfs.append(compute_user_feature_is_active(df, 'user_online_coupon'))
    df = pd.concat(feature_dfs, axis=1)
    return df


def extract_merchant_feature_offline(df, df_name):
    dfs = {
        'received_coupon': df[df.date.isnull() & df.coupon_id.notnull()],
        'buy_with_coupon': df[df.date.notnull() & df.coupon_id.notnull()],
        'buy_without_coupon': df[df.date.notnull() & df.coupon_id.isnull()],
    }
    df_received_coupon = dfs['received_coupon']
    df_buy_with_coupon = dfs['buy_with_coupon']
    df_buy_without_coupon = dfs['buy_without_coupon']
    feature_dfs = [
        ('received_coupon', compute_merchant_feature_coupon(df_received_coupon)),
        ('buy_with_coupon', compute_merchant_feature_coupon(df_buy_with_coupon)),
        ('buy_with_coupon', compute_merchant_feature_coupon_hold_days(df_buy_with_coupon)),
        ('received_coupon', compute_merchant_feature_date(df_received_coupon, 'date_received')),
        ('buy_with_coupon', compute_merchant_feature_date(df_buy_with_coupon, 'date_received')),
        ('buy_with_coupon', compute_merchant_feature_date(df_buy_with_coupon, 'date')),
        ('buy_without_coupon', compute_merchant_feature_date(df_buy_without_coupon, 'date')),
        ('buy_with_coupon', compute_merchant_feature_distance(df_buy_with_coupon)),
        ('buy_without_coupon', compute_merchant_feature_distance(df_buy_without_coupon)),
        ('received_coupon', compute_merchant_feature_to_user(df_received_coupon)),
        ('buy_with_coupon', compute_merchant_feature_to_user(df_buy_with_coupon)),
        ('buy_without_coupon', compute_merchant_feature_to_user(df_buy_without_coupon)),
    ]
    results = []
    for prefix, df_feat in feature_dfs:
        prefix_columns(df_feat, prefix)
        results.append(df_feat)
    df_result = pd.concat(results, axis=1)
    df_transfer_rate = compute_transfer_rate(
        df_result, 'received_coupon_coupon_count', 'buy_with_coupon_coupon_count')
    df_result = pd.concat([df_result, df_transfer_rate], axis=1)
    prefix_columns(df_result, df_name)
    return df_result


def extract_merchant_feature(dataset):
    return extract_merchant_feature_offline(dataset.offline, 'merchant_offline')


def extract_coupon_feature_offline(df, df_name):
    dfs = {
        'received_coupon': df[df.date.isnull() & df.coupon_id.notnull()],
        'buy_with_coupon': df[df.date.notnull() & df.coupon_id.notnull()],
    }
    feature_dfs = []
    for name, df_sub in dfs.items():
        df_sub = compute_coupon_feature_to_user(df_sub)
        prefix_columns(df_sub, name)
        feature_dfs.append(df_sub)
    df_result = pd.concat(feature_dfs, axis=1, sort=False)
    prefix_columns(df_result, df_name)
    return df_result


def extract_coupon_feature(dataset):
    return extract_coupon_feature_offline(dataset.offline, 'coupon_offline')


def extract_distance_feature_offline(df, df_name):
    dfs = {
        'received_coupon': df[df.date.isnull() & df.coupon_id.notnull()],
        'buy_with_coupon': df[df.date.notnull() & df.coupon_id.notnull()],
    }
    feature_dfs = []
    for name, df_sub in dfs.items():
        df_sub = compute_distance_feature_to_user(df_sub)
        prefix_columns(df_sub, name)
        feature_dfs.append(df_sub)
    df_result = pd.concat(feature_dfs, axis=1)
    prefix_columns(df_result, df_name)
    return df_result


def extract_distance_feature(dataset):
    return extract_distance_feature_offline(dataset.offline, 'distance_offline')


def extract_discount_feature(dataset):
    df_offline = compute_discount_feature(dataset.offline, 'discount_offline')
    df_online = compute_discount_feature(dataset.online_coupon, 'discount_online')
    df = pd.concat([df_offline, df_online], axis=1, sort=False)
    return df


def read_feature_dataset(num):
    dataset_paths = {
        'offline': f'data/z2_split_offline_feature_{num}.msgpack',
        'online_click': f'data/z2_split_online_click_feature_{num}.msgpack',
        'online_coupon': f'data/z2_split_online_coupon_feature_{num}.msgpack',
    }
    dataset = {}
    for name, path in dataset_paths.items():
        LOG.info('read dataset#{} {} -> {}', num, name, path)
        df = pd.read_msgpack(path)
        if 'distance' in df.columns:
            df['distance'].fillna(NULL_DISTANCE, inplace=True)
        dataset[name] = df
    dataset = FeatureDataset(**dataset)
    return dataset


def read_label_df(num):
    path = f'data/z2_split_label_{num}.msgpack'
    LOG.info('read label#{} -> {}', num, path)
    return pd.read_msgpack(path)


def extract_features(dataset, num):
    feature_extracts = {
        'user': extract_user_feature,
        'merchant': extract_merchant_feature,
        'coupon': extract_coupon_feature,
        'discount': extract_discount_feature,
        'distance': extract_distance_feature,
    }
    for name, extract in feature_extracts.items():
        LOG.info('#{} extract {} feature', num, name)
        df = extract(dataset)
        LOG.info('feature {}: size={}, columns={}', name, len(df), len(df.columns))
        df.to_msgpack('data/z3_feature_{}_{}.msgpack'.format(name, num))


def extract_labels(df, num):
    df = compute_predict_feature(df)
    LOG.info('#{} label: size={}, columns={}', num, len(df), len(df.columns))
    df.to_msgpack('data/z3_feature_label_{}.msgpack'.format(num))


def main():
    for num in [1, 2, 3]:
        dataset = read_feature_dataset(num)
        extract_features(dataset, num)
    for num in [1, 2]:
        df = read_label_df(num)
        extract_labels(df, num)
    df = pd.read_msgpack('data/z1_raw_test.msgpack')
    extract_labels(df, 3)


if __name__ == "__main__":
    main()
