import numpy as np
import pandas as pd
from loguru import logger as LOG
from pandas_parallel import pandas_parallel
from pandas_parallel import cloudpickle_safe_lru_cache


DATE_FORMAT = "%Y%m%d"
DAYS_15 = pd.Timedelta(15, unit='d')


@cloudpickle_safe_lru_cache()
def parse_discount_rate(discount_rate):
    """name, is_xianshi, is_dazhe, is_manjian, man, jian, rate"""
    if pd.isnull(discount_rate):
        return [np.nan, False, False, False, np.nan, np.nan, np.nan]
    if discount_rate == 'fixed':
        return ['fixed', True, False, False, np.nan, np.nan, np.nan]
    infos = discount_rate.split(':')
    is_manjian = len(infos) == 2
    if is_manjian:
        man = int(infos[0])
        jian = int(infos[1])
        rate = 1 - jian / man
        name = f'{man}:{jian}'
        return [name, False, False, True, man, jian, rate]
    else:
        man = 0
        jian = np.nan
        rate = float(infos[0])
        if rate > 1.0:
            rate = rate / 100
        name = str(int(rate * 100))
        return [name, False, True, False, man, jian, rate]


def apply_discount_rate(series):
    s = series.apply(parse_discount_rate)
    return pd.DataFrame.from_records(s.tolist(), index=s.index)


def read_raw_offline(filepath):
    """
    User_id,Merchant_id,Coupon_id,Discount_rate,Distance,Date_received,Date
    """
    dtype = {
        "User_id": np.int,
        "Merchant_id": np.int,
        "Coupon_id": np.str,
        "Discount_rate": np.str,
        "Distance": np.float,
        "Date_received": np.str,
        "Date": np.str,
    }
    na_values = "null"
    df = pd.read_csv(
        filepath, dtype=dtype, na_values=na_values, keep_default_na=False)
    return df


@pandas_parallel(progress_bar=True)
def parse_raw_offline(df):
    df['Date_received'] = pd.to_datetime(df['Date_received'], format=DATE_FORMAT)
    df['Date'] = pd.to_datetime(df['Date'], format=DATE_FORMAT)
    df_discount = apply_discount_rate(df['Discount_rate'])
    df_discount.drop(1, axis=1, inplace=True)
    df.drop('Discount_rate', axis=1, inplace=True)
    df = pd.concat([df, df_discount], axis=1)
    df.columns = [
        'user_id',
        'merchant_id',
        'coupon_id',
        'distance',
        'date_received',
        'date',
        'discount_name',
        'is_dazhe',
        'is_manjian',
        'discount_man',
        'discount_jian',
        'discount_rate',
    ]
    df['label'] = df['coupon_id'].notnull()\
        & df['date_received'].notnull()\
        & df['date'].notnull()\
        & (df['date'] - df['date_received'] <= DAYS_15)
    return df


def read_raw_online(filepath):
    """
    User_id,Merchant_id,Action,Coupon_id,Discount_rate,Date_received,Date
    """
    dtype = {
        "User_id": np.int,
        "Merchant_id": np.int,
        "Action": np.str,
        "Coupon_id": np.str,
        "Discount_rate": np.str,
        "Date_received": np.str,
        "Date": np.str,
    }
    na_values = "null"
    df = pd.read_csv(
        filepath, dtype=dtype, na_values=na_values, keep_default_na=False)
    return df


@pandas_parallel(progress_bar=True, scheduler='threads')
def parse_raw_online_click(df):
    df['Date_received'] = pd.to_datetime(df['Date_received'], format=DATE_FORMAT)
    df['Date'] = pd.to_datetime(df['Date'], format=DATE_FORMAT)
    df = df.groupby(['User_id', 'Merchant_id', 'Date']).size().to_frame('count')
    df = df.reset_index()
    df.columns = ['user_id', 'merchant_id', 'date', 'count']
    return df


ACTION_CLICK = '0'
ACTION_BUY = '1'
ACTION_RECEIVE = '2'


@pandas_parallel(progress_bar=True)
def parse_raw_online_coupon(df):
    df['Date_received'] = pd.to_datetime(df['Date_received'], format=DATE_FORMAT)
    df['Date'] = pd.to_datetime(df['Date'], format=DATE_FORMAT)
    df.drop('Action', axis=1, inplace=True)
    df_discount = apply_discount_rate(df['Discount_rate'])
    df.drop('Discount_rate', axis=1, inplace=True)
    df = pd.concat([df, df_discount], axis=1)
    df.columns = [
        'user_id',
        'merchant_id',
        'coupon_id',
        'date_received',
        'date',
        'discount_name',
        'is_xianshi',
        'is_dazhe',
        'is_manjian',
        'discount_man',
        'discount_jian',
        'discount_rate',
    ]
    return df


def split_raw_online(df):
    is_coupon = df["Action"] != ACTION_CLICK
    return df[~is_coupon].copy(), df[is_coupon].copy()


def read_raw_test(filepath):
    """
    User_id,Merchant_id,Coupon_id,Discount_rate,Distance,Date_received
    """
    dtype = {
        "User_id": np.int,
        "Merchant_id": np.int,
        "Coupon_id": np.str,
        "Discount_rate": np.str,
        "Distance": np.float,
        "Date_received": np.str,
    }
    na_values = "null"
    df = pd.read_csv(
        filepath, dtype=dtype, na_values=na_values, keep_default_na=False)
    return df


@pandas_parallel(progress_bar=True)
def parse_raw_test(df):
    df['Date_received'] = pd.to_datetime(df['Date_received'], format=DATE_FORMAT)
    df_discount = apply_discount_rate(df['Discount_rate'])
    df_discount.drop(1, axis=1, inplace=True)
    df.drop('Discount_rate', axis=1, inplace=True)
    df = pd.concat([df, df_discount], axis=1)
    df.columns = [
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
    ]
    return df


def main():
    LOG.info('read_raw_offline')
    df = read_raw_offline('data/ccf_offline_stage1_train.csv')
    LOG.info('parse_raw_offline')
    df = parse_raw_offline(df)
    df.info()
    df.to_msgpack('data/z1_raw_offline.msgpack')

    LOG.info('read_raw_online')
    df = read_raw_online('data/ccf_online_stage1_train.csv')
    LOG.info('split_raw_online')
    df_click, df_coupon = split_raw_online(df)

    LOG.info('parse_raw_online_click')
    df_click = parse_raw_online_click(df_click)
    df_click.info()
    df_click.to_msgpack('data/z1_raw_online_click.msgpack')

    LOG.info('parse_raw_online_coupon')
    df_coupon = parse_raw_online_coupon(df_coupon)
    df_coupon.info()
    df_coupon.to_msgpack('data/z1_raw_online_coupon.msgpack')

    LOG.info('read_raw_test')
    df = read_raw_test('data/ccf_offline_stage1_test_revised.csv')
    LOG.info('parse_raw_test')
    df = parse_raw_test(df)
    df.info()
    df.to_msgpack('data/z1_raw_test.msgpack')


if __name__ == '__main__':
    main()
