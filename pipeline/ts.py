"""
User(user_id):
    offline_coupons:
        - date_received
        - date
        - coupon_id
        - merchant_id
        - discount
        - distance
    offline_buys:
        - merchant_id
        - distance
    online_coupons:
        -
    online_buys:
        -
    online_click:
        -

Merchant(merchant_id):
    offline_coupons:
        - date_received
        - date
        - coupon_id
        - user_id
        - discount
        - distance
    offline_buys:
        - merchant_id
        - distance

Coupon(coupon_id):
    offline_coupons:
        - date_received
        - date
        - coupon_id
        - user_id
        - discount
        - distance

Discount(discount_name):
    -


Event:
    - date
    - type
        - offline_receive_coupon OFF_RC
        - offline_buy_with_coupon OFF_BC
        - offline_buy_without_coupon OFF_B
        - online_receive_coupon ON_RC
        - online_buy_with_coupon ON_BC
        - online_buy_without_coupon ON_B
        - online_click ON_C
    - user_id
    - merchant_id
    - coupon_id
    - discount
    - distance
    - click_count

Discount:
    - name
    - type
    - man
    - jian
    - rate


UserEvents:

{
    user_id: [
        [date, {
            event_type: {

            }
        }]
    ]
}
"""
import time
from collections import OrderedDict
from contextlib import contextmanager

import numpy as np
import pandas as pd
import tqdm
from loguru import logger as LOG
from cached_property import cached_property

from z1_raw import apply_discount_rate


@contextmanager
def TLOG(name):
    t0 = time.time()
    try:
        yield
    finally:
        t1 = time.time()
        LOG.info('{} cost {:.3f}s', name, t1 - t0)

# Event types:
#   offline_receive_coupon
#   offline_buy_with_coupon
#   offline_buy_without_coupon
#   online_receive_coupon
#   online_buy_with_coupon
#   online_buy_without_coupon
#   online_click


EVENT_COLUMNS = [
    'date',
    'date2',
    'event_type',
    'user_id',
    'merchant_id',
    'coupon_id',
    'discount_name',
    'distance',
    'click_count',
]


def _get_coupon_event_type(df, line):
    buy_without_coupon = df['coupon_id'].isnull() | df['date_received'].isnull()
    buy_with_coupon = (~buy_without_coupon) & df['date'].notnull()
    receive_coupon = (~buy_without_coupon) & df['date'].isnull()
    event_type = pd.concat([
        pd.Series(line + '_buy_with_coupon', index=df.index[buy_with_coupon]),
        pd.Series(line + '_buy_without_coupon', index=df.index[buy_without_coupon]),
        pd.Series(line + '_receive_coupon', index=df.index[receive_coupon]),
    ]).sort_index()
    return event_type


class O2OEvents:
    def __init__(self, user_id_index):
        self._data = []
        self.user_id_index = user_id_index

    def _feed_coupon(self, df, line):
        event_type = _get_coupon_event_type(df, line)
        is_receive_coupon = event_type == line + '_receive_coupon'
        is_buy_with_coupon = event_type == line + '_buy_with_coupon'
        df_receive_coupon = df.loc[is_receive_coupon].copy()
        df_buy_with_coupon = df.loc[is_buy_with_coupon].copy()
        df_buy_without_coupon = df.loc[~(is_receive_coupon | is_buy_with_coupon)].copy()

        # date:事件发生时间，date2:相关的一个时间
        df_receive_coupon['event_type'] = line + '_receive_coupon'
        df_receive_coupon['date2'] = df_receive_coupon['date']
        df_receive_coupon['date'] = df_receive_coupon['date_received']

        # buy_with_coupon 包含两个事件，一个领券一个用券
        df_receive_coupon2 = df_buy_with_coupon.copy()
        df_receive_coupon2['event_type'] = line + '_receive_coupon'
        df_receive_coupon2['date2'] = df_receive_coupon2['date']
        df_receive_coupon2['date'] = df_receive_coupon2['date_received']

        df_buy_with_coupon['event_type'] = line + '_buy_with_coupon'
        df_buy_with_coupon['date2'] = df_buy_with_coupon['date_received']

        df_buy_without_coupon['event_type'] = line + '_buy_without_coupon'
        df_buy_without_coupon['date2'] = pd.NaT

        df = pd.concat([
            df_receive_coupon,
            df_receive_coupon2,
            df_buy_with_coupon,
            df_buy_without_coupon,
        ], ignore_index=True)

        if line == 'online':
            df['distance'] = np.NAN
        df['click_count'] = np.NAN
        df_result = df[[
            'date', 'date2', 'event_type', 'user_id', 'merchant_id',
            'coupon_id', 'discount_name', 'distance', 'click_count',
        ]].reset_index(drop=True)
        self._data.append(df_result)

    def feed_offline(self, df):
        self._feed_coupon(df, 'offline')

    def feed_online_coupon(self, df):
        df = df[df['user_id'].isin(self.user_id_index)]
        self._feed_coupon(df, 'online')

    def feed_online_click(self, df):
        df = df[df['user_id'].isin(self.user_id_index)]
        result = OrderedDict()
        result['date'] = df['date']
        result['date2'] = pd.NaT
        result['event_type'] = 'online_click'
        result['user_id'] = df['user_id']
        result['merchant_id'] = df['merchant_id']
        result['coupon_id'] = np.NAN
        result['discount_name'] = np.NAN
        result['distance'] = np.NAN
        result['click_count'] = df['count']
        df_result = pd.DataFrame.from_dict(result)
        self._data.append(df_result)

    def feed_test(self, df):
        result = OrderedDict()
        result['date'] = df['date_received']
        result['date2'] = pd.NaT
        result['event_type'] = 'offline_receive_coupon'
        result['user_id'] = df['user_id']
        result['merchant_id'] = df['merchant_id']
        result['coupon_id'] = df['coupon_id']
        result['discount_name'] = df['discount_name']
        result['distance'] = df['distance']
        result['click_count'] = np.NAN
        df_result = pd.DataFrame.from_dict(result)
        self._data.append(df_result)

    def to_frame(self):
        df = pd.concat(self._data, ignore_index=True).reset_index(drop=True)
        df['event_type'] = df['event_type'].astype('category')
        df['discount_name'] = df['discount_name'].astype('category')
        df.sort_values(['date', 'event_type'], inplace=True)
        return df

    @staticmethod
    def main():
        df_raw_offline = pd.read_msgpack(f'data/z1_raw_offline.msgpack')
        df_raw_test = pd.read_msgpack(f'data/z1_raw_test.msgpack')

        LOG.info('build_discount_table')
        df_discount = build_discount_table(df_raw_offline, df_raw_test)
        df_discount.to_msgpack('data/z6_ts_discount.msgpack')

        LOG.info('build_index_of user_id')
        user_id_index = build_index_of(df_raw_offline, df_raw_test, 'user_id')
        np.save('data/z6_ts_user_id.npy', user_id_index)

        for key in ['merchant_id', 'coupon_id']:
            LOG.info('build_index_of {}', key)
            arr = build_index_of(df_raw_offline, df_raw_test, key)
            np.save('data/z6_ts_{}.npy'.format(key), arr)

        LOG.info('build_index_of user_id_merchant_id')
        arr = build_index_of(df_raw_offline, df_raw_test, ['user_id', 'merchant_id'])
        np.save('data/z6_ts_user_id_merchant_id.npy', arr)

        events = O2OEvents(user_id_index)
        LOG.info('events feed_offline')
        events.feed_offline(df_raw_offline)
        df_online_coupon = pd.read_msgpack(f'data/z1_raw_online_coupon.msgpack')
        LOG.info('events feed_online_coupon')
        events.feed_online_coupon(df_online_coupon)
        df_online_click = pd.read_msgpack(f'data/z1_raw_online_click.msgpack')
        LOG.info('events feed_online_click')
        events.feed_online_click(df_online_click)
        LOG.info('events feed_test')
        events.feed_test(df_raw_test)
        LOG.info('events to_frame')
        df = events.to_frame()
        df.to_msgpack('data/z6_ts_events.msgpack')


def build_discount_table(df_events, df_raw_test):
    discounts = []
    for df in [df_events, df_raw_test]:
        discounts.append(np.array(df['discount_name'].dropna().unique()))
    discounts = pd.Series(np.unique(np.concatenate(discounts)))
    discount_table = apply_discount_rate(discounts)
    discount_table.columns = [
        'discount_name',
        'is_xianshi',
        'is_dazhe',
        'is_manjian',
        'discount_man',
        'discount_jian',
        'discount_rate',
    ]
    discount_table['discount_name'] = discount_table['discount_name'].astype('category')
    discount_table.set_index('discount_name', inplace=True)
    discount_table.sort_index(inplace=True)
    discount_table.sort_values
    return discount_table


def build_index_of(df_raw_offline, df_raw_test, key):
    values = []
    for df in [df_raw_offline, df_raw_test]:
        if isinstance(key, (tuple, list)):
            v = pd.unique(list(df[key].dropna().itertuples(index=False, name=None)))
        else:
            v = np.array(df[key].dropna().unique())
        values.append(v)
    return np.unique(np.concatenate(values))


def set_and_sort_index(df, key):
    if isinstance(key, (tuple, list)):
        index_columns = [*key, 'date', 'event_type']
        df = df.loc[df[key].dropna().index]
    else:
        index_columns = [key, 'date', 'event_type']
        df = df[df[key].notnull()]
    df = df.set_index(index_columns)
    df.sort_index(inplace=True)
    return df


def where_event_type(df, event_type):
    if isinstance(event_type, (list, tuple)):
        return df.loc[pd.IndexSlice[:, event_type], :]
    try:
        return df.xs(event_type, level='event_type')
    except KeyError:
        # return empty dataframe instead of raise KeyError
        return df.head(0)


class EventsSnapshot:
    def __init__(self, df, date, recent_begin, recent_end):
        self.df = df.loc[:recent_end]
        self.date = date
        self.recent_begin = recent_begin
        self.recent_end = recent_end

    def __repr__(self):
        return '<{} {}/{}>'.format(
            self.__class__.__name__, self.recent_begin.date(), self.recent_end.date())

    @cached_property
    def history(self):
        return self.df.loc[:self.recent_begin].copy()

    @cached_property
    def recent(self):
        return self.df.loc[self.recent_begin:self.recent_end].copy()

    @cached_property
    def recent_offline(self):
        offline = ['offline_receive_coupon',
                   'offline_buy_with_coupon',
                   'offline_buy_without_coupon']
        return where_event_type(self.recent, offline)

    @cached_property
    def recent_offline_receive_coupon(self):
        return where_event_type(self.recent, 'offline_receive_coupon')

    @cached_property
    def recent_offline_buy_with_coupon(self):
        return where_event_type(self.recent, 'offline_buy_with_coupon')

    @cached_property
    def recent_offline_buy_without_coupon(self):
        return where_event_type(self.recent, 'offline_buy_without_coupon')

    @cached_property
    def recent_online_receive_coupon(self):
        return where_event_type(self.recent, 'online_receive_coupon')

    @cached_property
    def recent_online_buy_with_coupon(self):
        return where_event_type(self.recent, 'online_buy_with_coupon')

    @cached_property
    def recent_online_buy_without_coupon(self):
        return where_event_type(self.recent, 'online_buy_without_coupon')

    @cached_property
    def recent_online_click(self):
        return where_event_type(self.recent, 'online_click')

    @cached_property
    def history_offline_receive_coupon(self):
        return where_event_type(self.history, 'offline_receive_coupon')

    @cached_property
    def history_offline_buy_with_coupon(self):
        return where_event_type(self.history, 'offline_buy_with_coupon')

    @cached_property
    def history_offline_buy_without_coupon(self):
        return where_event_type(self.history, 'offline_buy_without_coupon')

    @cached_property
    def history_online_receive_coupon(self):
        return where_event_type(self.history, 'online_receive_coupon')

    @cached_property
    def history_online_buy_with_coupon(self):
        return where_event_type(self.history, 'online_buy_with_coupon')

    @cached_property
    def history_online_buy_without_coupon(self):
        return where_event_type(self.history, 'online_buy_without_coupon')

    @cached_property
    def history_online_click(self):
        return where_event_type(self.history, 'online_click')


def divide_zero(a, b):
    if np.isclose(b, 0):
        return 0
    return a / b


def with_discount(df, df_discount):
    return pd.merge(df, df_discount, how='left', left_on='discount_name', right_index=True)


class BaseFeature:

    def __init__(self, df_events, keys, df_discount, recent_days=60, recent_offset_days=15):
        self.name = self.__class__.__name__
        self.columns = self.__class__.COLUMNS
        if not isinstance(self.__class__.KEY_NAMES, (tuple, list)):
            self.key_names = [self.__class__.KEY_NAMES]
        else:
            self.key_names = list(self.__class__.KEY_NAMES)
        self.is_multikey = len(self.key_names) > 1
        self.df_events = df_events
        self.keys = keys
        self.df_discount = df_discount
        self.recent_days = recent_days
        self.recent_offset_days = recent_offset_days
        self._recent_offset_timedelta = pd.Timedelta(days=self.recent_offset_days)
        self._recent_timedelta = pd.Timedelta(days=self.recent_days - 1)
        with TLOG('preprocess events'):
            self.df = self.preprocess_events(df_events, self.key_names)

    def preprocess_events(self, df, key_names):
        index_columns = [*key_names, 'date', 'event_type']
        if self.is_multikey:
            df = df.loc[df[key_names].dropna().index]
        else:
            df = df[df[key_names[0]].notnull()]
        df = df.set_index(index_columns)
        df.sort_index(inplace=True)
        return df

    def date_span_of(self, value):
        df_receive_coupon = value.xs('offline_receive_coupon', level='event_type')
        idx_date = df_receive_coupon.index.get_level_values('date').unique()
        recent_end = idx_date - self._recent_offset_timedelta
        recent_begin = recent_end - self._recent_timedelta
        return zip(idx_date, recent_begin, recent_end)

    def with_discount(self, df):
        return with_discount(df, self.df_discount)

    def process_snapshot(self, key, snapshot):
        raise NotImplementedError

    def get_value(self, key):
        if self.is_multikey:
            return self.df.loc[tuple(key)]
        else:
            return self.df.loc[key]

    def process(self):
        result = []
        for key in tqdm.tqdm(self.keys, desc=self.name):
            df_sub = self.get_value(key).copy()
            for date, recent_begin, recent_end in self.date_span_of(df_sub):
                snapshot = EventsSnapshot(df_sub, date, recent_begin, recent_end)
                if self.is_multikey:
                    row = self.process_snapshot(*key, date, snapshot)
                    result.append([*key, date, *row])
                else:
                    row = self.process_snapshot(key, date, snapshot)
                    result.append([key, date, *row])
        df_result = pd.DataFrame.from_records(result, columns=self.columns)
        return df_result


class UserFeature(BaseFeature):

    KEY_NAMES = 'user_id'
    COLUMNS = [
        'user_id',
        'date',
        'recent_receive_coupon_count',
        'recent_buy_with_coupon_count',
        'recent_hold_coupon_count',
        'recent_buy_with_coupon_rate',
        'recent_buy_with_man_200_rate',
        'recent_buy_with_man_200_share',
        'recent_discount_rate_mean',
        'recent_discount_rate_min',
        'recent_discount_rate_max',
        'recent_buy_with_coupon_merchant_nunique',
        'recent_buy_with_coupon_merchant_nunique_share',
        'recent_buy_with_coupon_coupon_nunique',
        'recent_buy_with_coupon_coupon_nunique_share',
        'recent_buy_with_coupon_mean_count_per_merchant',
        'recent_buy_with_coupon_distance_mean',
        'recent_buy_with_coupon_distance_min',
        'recent_buy_with_coupon_distance_max',
        'recent_online_click_count',
        'recent_online_receive_coupon_count',
        'recent_online_buy_without_coupon_count',
        'recent_online_buy_with_coupon_count',
        'recent_online_not_buy_coupon_count',
        'recent_online_buy_with_coupon_rate',
        'recent_online_receive_coupon_rate',
    ]

    def process_snapshot(self, user_id, date, snapshot):
        return [
            *self.process_offline(user_id, date, snapshot),
            *self.process_online(user_id, date, snapshot),
        ]

    def process_offline(self, user_id, date, snapshot):
        # 用户领取优惠券次数
        recent_receive_coupon_count = len(snapshot.recent_offline_receive_coupon)
        # 用户获得优惠券并核销次数
        recent_buy_with_coupon_count = len(snapshot.recent_offline_buy_with_coupon)
        # 用户获得优惠券但没有消费的次数
        recent_hold_coupon_count = recent_receive_coupon_count - recent_buy_with_coupon_count
        # 用户领取优惠券后进行核销率
        recent_buy_with_coupon_rate = divide_zero(
            recent_buy_with_coupon_count, recent_receive_coupon_count)
        # 用户满200~500减的优惠券核销率
        recent_offline_receive_coupon = self.with_discount(
            snapshot.recent_offline_receive_coupon)
        recent_receive_man_200_count = len(
            recent_offline_receive_coupon[recent_offline_receive_coupon['discount_man'] >= 200])
        recent_offline_buy_with_coupon = self.with_discount(
            snapshot.recent_offline_buy_with_coupon)
        recent_buy_with_man_200_count = len(
            recent_offline_buy_with_coupon[recent_offline_buy_with_coupon['discount_man'] >= 200])
        recent_buy_with_man_200_rate = divide_zero(
            recent_buy_with_man_200_count, recent_receive_man_200_count)
        # 用户核销满200~500减的优惠券占所有核销优惠券的比重
        recent_buy_with_man_200_share = divide_zero(
            recent_buy_with_man_200_count, recent_buy_with_coupon_count)
        # 用户核销优惠券的平均/最低/最高消费折率
        discount_rate_stats = recent_offline_buy_with_coupon['discount_rate']\
            .agg(['mean', 'min', 'max'])
        # 用户核销过优惠券的不同商家数量，及其占所有不同商家的比重
        buy_with_coupon_merchant_nunique = recent_offline_buy_with_coupon['merchant_id'].nunique()
        recent_offline = self.with_discount(snapshot.recent_offline)
        merchant_nunique = recent_offline['merchant_id'].nunique()
        buy_with_coupon_merchant_nunique_share = divide_zero(
            buy_with_coupon_merchant_nunique, merchant_nunique)
        # 用户核销过的不同优惠券数量，及其占所有不同优惠券的比重
        buy_with_coupon_coupon_nunique = recent_offline_buy_with_coupon['coupon_id'].nunique()
        coupon_nunique = recent_offline['coupon_id'].nunique()
        buy_with_coupon_coupon_nunique_share = divide_zero(
            buy_with_coupon_coupon_nunique, coupon_nunique)
        # 用户平均核销每个商家多少张优惠券
        buy_with_coupon_mean_count_per_merchant = recent_offline_buy_with_coupon\
            .groupby('merchant_id').size().mean()
        # 用户核销优惠券中的平均/最大/最小用户-商家距离
        buy_with_coupon_distance_stats = recent_offline_buy_with_coupon['distance']\
            .agg(['mean', 'min', 'max'])
        return [
            recent_receive_coupon_count,
            recent_buy_with_coupon_count,
            recent_hold_coupon_count,
            recent_buy_with_coupon_rate,
            recent_buy_with_man_200_rate,
            recent_buy_with_man_200_share,
            *discount_rate_stats,
            buy_with_coupon_merchant_nunique,
            buy_with_coupon_merchant_nunique_share,
            buy_with_coupon_coupon_nunique,
            buy_with_coupon_coupon_nunique_share,
            buy_with_coupon_mean_count_per_merchant,
            *buy_with_coupon_distance_stats,
        ]

    def process_online(self, user_id, date, snapshot):
        # 用户线上点击次数
        recent_click_count = len(snapshot.recent_online_click)
        # 用户线上领取次数
        recent_receive_coupon_count = len(snapshot.recent_online_receive_coupon)
        # 用户线上无券购买次数
        recent_buy_without_coupon_count = len(snapshot.recent_online_buy_without_coupon)
        # 用户线上核销次数
        recent_buy_with_coupon_count = len(snapshot.recent_online_buy_with_coupon)
        # 用户线上不消费次数
        recent_not_buy_coupon_count = recent_receive_coupon_count - recent_buy_with_coupon_count
        # 用户线上优惠券核销率
        recent_buy_with_coupon_rate = divide_zero(
            recent_buy_with_coupon_count, recent_receive_coupon_count)
        # 用户线上优惠券领取率
        recent_receive_coupon_rate = divide_zero(recent_receive_coupon_count, recent_click_count)
        return [
            recent_click_count,
            recent_receive_coupon_count,
            recent_buy_without_coupon_count,
            recent_buy_with_coupon_count,
            recent_not_buy_coupon_count,
            recent_buy_with_coupon_rate,
            recent_receive_coupon_rate,
        ]

    @staticmethod
    def main():
        with TLOG('read data'):
            df = pd.read_msgpack('data/z6_ts_events.msgpack')
            df_discount = pd.read_msgpack('data/z6_ts_discount.msgpack')
            user_ids = np.load('data/z6_ts_user_id.npy')
        user_feature = UserFeature(df, df_discount=df_discount, keys=user_ids)
        df = user_feature.process()
        df.to_msgpack('data/z6_ts_feature_user.msgpack')


class MerchantFeature(BaseFeature):

    KEY_NAMES = 'merchant_id'
    COLUMNS = [
        'merchant_id',
        'date',
        'recent_receive_coupon_count',
        'recent_buy_with_coupon_count',
        'recent_not_buy_coupon_count',
        'recent_buy_with_coupon_rate',
        'buy_with_coupon_coupon_discount_mean',
        'buy_with_coupon_coupon_discount_min',
        'buy_with_coupon_coupon_discount_max',
        'buy_with_coupon_user_nunique',
        'buy_with_coupon_user_share',
        'buy_with_coupon_mean_per_user',
        'buy_with_coupon_coupon_nunique',
        'buy_with_coupon_coupon_share',
        'buy_with_coupon_mean_per_coupon',
        'buy_with_coupon_mean_days',
        'buy_with_coupon_distance_mean',
        'buy_with_coupon_distance_min',
        'buy_with_coupon_distance_max',
    ]

    def process_snapshot(self, merchant_id, date, snapshot):
        # 商家优惠券被领取次数
        recent_receive_coupon_count = len(snapshot.recent_offline_receive_coupon)
        # 商家优惠券被领取后核销次数
        recent_buy_with_coupon_count = len(snapshot.recent_offline_buy_with_coupon)
        # 商家优惠券被领取后不核销次数
        recent_not_buy_coupon_count = recent_receive_coupon_count - recent_buy_with_coupon_count
        # 商家优惠券被领取后核销率
        recent_buy_with_coupon_rate = divide_zero(
            recent_buy_with_coupon_count, recent_receive_coupon_count)
        # 商家优惠券核销的平均/最小/最大消费折率
        buy_with_coupon = self.with_discount(snapshot.recent_offline_buy_with_coupon)
        buy_with_coupon_coupon_stats = buy_with_coupon['discount_rate'].agg(['mean', 'min', 'max'])
        # 核销商家优惠券的不同用户数量，及其占领取不同的用户比重
        buy_with_coupon_user_nunique = buy_with_coupon['user_id'].nunique()
        receive_coupon_user_nunique = snapshot.recent_offline_receive_coupon['user_id'].nunique()
        buy_with_coupon_user_share = divide_zero(
            buy_with_coupon_user_nunique, receive_coupon_user_nunique)
        # 商家优惠券平均每个用户核销多少张
        buy_with_coupon_mean_per_user = buy_with_coupon.groupby('user_id').size().mean()
        # 商家被核销过的不同优惠券数量
        buy_with_coupon_coupon_nunique = buy_with_coupon['coupon_id'].nunique()
        # 商家被核销过的不同优惠券数量占所有领取过的不同优惠券数量的比重
        receive_coupon_coupon_nunique = snapshot.recent_offline_receive_coupon['coupon_id'].nunique(
        )
        buy_with_coupon_coupon_share = divide_zero(
            buy_with_coupon_coupon_nunique, receive_coupon_coupon_nunique)
        # 商家平均每种优惠券核销多少张
        buy_with_coupon_mean_per_coupon = buy_with_coupon.groupby('coupon_id').size().mean()
        # 商家被核销优惠券的平均时间率
        buy_with_coupon_days = (
            pd.Series(buy_with_coupon.index.get_level_values('date')) -
            buy_with_coupon['date2'].reset_index(drop=True)
        ).map(lambda x: x.days)
        buy_with_coupon_mean_days = buy_with_coupon_days.mean()
        # 商家被核销优惠券中的平均/最小/最大用户-商家距离
        buy_with_coupon_distance_stats = buy_with_coupon['distance']\
            .agg(['mean', 'min', 'max'])
        return [
            recent_receive_coupon_count,
            recent_buy_with_coupon_count,
            recent_not_buy_coupon_count,
            recent_buy_with_coupon_rate,
            *buy_with_coupon_coupon_stats,
            buy_with_coupon_user_nunique,
            buy_with_coupon_user_share,
            buy_with_coupon_mean_per_user,
            buy_with_coupon_coupon_nunique,
            buy_with_coupon_coupon_share,
            buy_with_coupon_mean_per_coupon,
            buy_with_coupon_mean_days,
            *buy_with_coupon_distance_stats,
        ]

    @staticmethod
    def main():
        with TLOG('read data'):
            df = pd.read_msgpack('data/z6_ts_events.msgpack')
            df_discount = pd.read_msgpack('data/z6_ts_discount.msgpack')
            merchant_ids = np.load('data/z6_ts_merchant_id.npy')
        merchant_feature = MerchantFeature(df, df_discount=df_discount, keys=merchant_ids)
        df = merchant_feature.process()
        df.to_msgpack('data/z6_ts_feature_merchant.msgpack')


class UserMerchantFeature(BaseFeature):

    KEY_NAMES = ['user_id', 'merchant_id']
    COLUMNS = [
        'user_id',
        'merchant_id',
        'date',
        'recent_receive_coupon_count',
        'recent_buy_with_coupon_count',
        'recent_not_buy_coupon_count',
        'recent_buy_with_coupon_rate',
        'recent_not_buy_share_user',
        'recent_buy_share_user',
        'recent_not_buy_share_merchant',
        'recent_buy_share_merchant',
    ]

    def __init__(self, *args, df_feature_user, df_feature_merchant, **kwargs):
        super().__init__(*args, **kwargs)
        self.df_feature_user = df_feature_user.set_index(['user_id', 'date'])
        self.df_feature_merchant = df_feature_merchant.set_index(['merchant_id', 'date'])

    def process_snapshot(self, user_id, merchant_id, date, snapshot):
        # 用户领取商家的优惠券次数
        recent_receive_coupon_count = len(snapshot.recent_offline_receive_coupon)
        # 用户领取商家的优惠券后核销次数
        recent_buy_with_coupon_count = len(snapshot.recent_offline_buy_with_coupon)
        # 用户领取商家的优惠券后不核销次数
        recent_not_buy_coupon_count = recent_receive_coupon_count - recent_buy_with_coupon_count
        # 用户领取商家的优惠券后核销率
        recent_buy_with_coupon_rate = divide_zero(
            recent_buy_with_coupon_count, recent_receive_coupon_count)
        # 用户对商家的不核销次数占用户总的不核销次数的比重
        user_feature = self.df_feature_user.loc[user_id, date]
        user_not_buy_coupon_count = user_feature['recent_hold_coupon_count']
        not_buy_share_user = divide_zero(recent_not_buy_coupon_count, user_not_buy_coupon_count)
        # 用户对每个商家的优惠券核销次数占用户总的核销次数的比重
        user_buy_with_coupon_coupon = user_feature['recent_buy_with_coupon_count']
        buy_share_user = divide_zero(recent_buy_with_coupon_count, user_buy_with_coupon_coupon)
        # 用户对每个商家的不核销次数占商家总的不核销次数的比重
        merchant_feature = self.df_feature_merchant.loc[merchant_id, date]
        merchant_not_buy_coupon_count = merchant_feature['recent_not_buy_coupon_count']
        not_buy_share_merchant = divide_zero(
            recent_not_buy_coupon_count, merchant_not_buy_coupon_count)
        # 用户对每个商家的优惠券核销次数占商家总的核销次数的比重
        merchant_buy_with_coupon_count = merchant_feature['recent_buy_with_coupon_count']
        buy_share_merchant = divide_zero(
            recent_buy_with_coupon_count, merchant_buy_with_coupon_count)
        return [
            recent_receive_coupon_count,
            recent_buy_with_coupon_count,
            recent_not_buy_coupon_count,
            recent_buy_with_coupon_rate,
            not_buy_share_user,
            buy_share_user,
            not_buy_share_merchant,
            buy_share_merchant,
        ]

    @staticmethod
    def main():
        with TLOG('read data'):
            df = pd.read_msgpack('data/z6_ts_events.msgpack')
            df_discount = pd.read_msgpack('data/z6_ts_discount.msgpack')
            ids = np.load('data/z6_ts_user_id_merchant_id.npy')
            df_feature_user = pd.read_msgpack('data/z6_ts_feature_user.msgpack')
            df_feature_merchant = pd.read_msgpack('data/z6_ts_feature_merchant.msgpack')
        feature = UserMerchantFeature(
            df, df_discount=df_discount, keys=ids,
            df_feature_user=df_feature_user, df_feature_merchant=df_feature_merchant)
        df = feature.process()
        df.to_msgpack('data/z6_ts_feature_user_merchant.msgpack')


class CouponFeature(BaseFeature):

    KEY_NAMES = 'coupon_id'
    COLUMNS = [
        'coupon_id',
        'date',
        'recent_receive_coupon_count',
        'recent_buy_with_coupon_count',
        'recent_buy_with_coupon_rate',
        'buy_with_coupon_user_nunique',
        'buy_with_coupon_mean_days',
        'buy_with_coupon_distance_mean',
        'buy_with_coupon_distance_min',
        'buy_with_coupon_distance_max',
    ]

    def process_snapshot(self, coupon_id, date, snapshot):
        # 领取次数
        recent_receive_coupon_count = len(snapshot.recent_offline_receive_coupon)
        # 领取后核销次数
        recent_buy_with_coupon_count = len(snapshot.recent_offline_buy_with_coupon)
        # 核销率
        recent_buy_with_coupon_rate = recent_buy_with_coupon_count / \
            (recent_receive_coupon_count + 1)
        # 领取该优惠券用户数
        buy_with_coupon = snapshot.recent_offline_buy_with_coupon
        buy_with_coupon_user_nunique = buy_with_coupon['user_id'].nunique()
        # 核销时间
        buy_with_coupon_days = (
            pd.Series(buy_with_coupon.index.get_level_values('date')) -
            buy_with_coupon['date2'].reset_index(drop=True)
        ).map(lambda x: x.days)
        buy_with_mean_days = buy_with_coupon_days.mean()
        # 核销优惠券中的平均/最小/最大用户-商家距离
        buy_with_coupon_distance_stats = buy_with_coupon['distance']\
            .agg(['mean', 'min', 'max'])
        return [
            recent_receive_coupon_count,
            recent_buy_with_coupon_count,
            recent_buy_with_coupon_rate,
            buy_with_coupon_user_nunique,
            buy_with_mean_days,
            *buy_with_coupon_distance_stats,
        ]

    @staticmethod
    def main():
        with TLOG('read data'):
            df = pd.read_msgpack('data/z6_ts_events.msgpack')
            df_discount = pd.read_msgpack('data/z6_ts_discount.msgpack')
            coupon_ids = np.load('data/z6_ts_coupon_id.npy')
        coupon_feature = CouponFeature(df, df_discount=df_discount, keys=coupon_ids)
        df = coupon_feature.process()
        df.to_msgpack('data/z6_ts_feature_coupon.msgpack')


class MergedFeature:

    @staticmethod
    def merge_feature(df, df_user, df_merchant, df_coupon, df_user_merchant):
        df = pd.merge(df, df_user, how='left', suffixes=['', '_user'],
                      left_on=['user_id', 'date'],
                      right_on=['user_id', 'date'])

        df = pd.merge(df, df_merchant, how='left', suffixes=['', '_merchant'],
                      left_on=['merchant_id', 'date'],
                      right_on=['merchant_id', 'date'])

        df = pd.merge(df, df_coupon, how='left', suffixes=['', '_coupon'],
                      left_on=['coupon_id', 'date'],
                      right_on=['coupon_id', 'date'])

        df = pd.merge(df, df_user_merchant, how='left', suffixes=['', '_user_merchant'],
                      left_on=['user_id', 'merchant_id', 'date'],
                      right_on=['user_id', 'merchant_id', 'date'])

        df = df.reset_index(drop=True)
        df_full = df.copy()

        df = df.drop(['user_id', 'merchant_id', 'coupon_id', 'discount_name'], axis=1)
        df.fillna(0, inplace=True)

        return df, df_full

    @staticmethod
    def main():
        LOG.info('read features')
        df_user = pd.read_msgpack('data/z6_ts_feature_user.msgpack')
        df_merchant = pd.read_msgpack('data/z6_ts_feature_merchant.msgpack')
        df_coupon = pd.read_msgpack('data/z6_ts_feature_coupon.msgpack')
        df_user_merchant = pd.read_msgpack('data/z6_ts_feature_user_merchant.msgpack')
        df_coupon = pd.read_msgpack('data/z6_ts_feature_coupon.msgpack')
        df_discount = pd.read_msgpack('data/z6_ts_discount.msgpack')
        df_events = pd.read_msgpack('data/z6_ts_events.msgpack')

        LOG.info('split train/test, compute train label')
        mask = df_events['event_type'] == 'offline_receive_coupon'
        df_train_events = df_events[mask & (df_events['date'] < '2016-07-01')].copy()
        df_train_events['label'] = (df_train_events['date2'] - df_train_events['date'])\
            .map(lambda x: x.days <= 15)
        df_test_events = df_events[mask & (df_events['date'] >= '2016-07-01')]
        df_test_events = df_test_events[df_test_events['event_type'] == 'offline_receive_coupon']
        columns = ['user_id', 'merchant_id', 'coupon_id', 'date', 'distance', 'discount_name']

        LOG.info('merge for test')
        df_test = df_test_events[columns]
        df, df_full = MergedFeature.merge_feature(
            with_discount(df_test, df_discount),
            df_user=df_user,
            df_merchant=df_merchant,
            df_coupon=df_coupon,
            df_user_merchant=df_user_merchant,
        )
        LOG.info('test shape={}, full_shape={}', df.shape, df_full.shape)
        df.to_msgpack('data/z6_ts_merged_test.msgpack')
        df_full.to_msgpack('data/z6_ts_merged_test_full.msgpack')

        LOG.info('merge for train')
        df_train = df_train_events[columns + ['label']]
        df, df_full = MergedFeature.merge_feature(
            with_discount(df_train, df_discount),
            df_user=df_user,
            df_merchant=df_merchant,
            df_coupon=df_coupon,
            df_user_merchant=df_user_merchant,
        )
        LOG.info('train shape={}, full_shape={}', df.shape, df_full.shape)
        df.to_msgpack('data/z6_ts_merged_train.msgpack')
        df_full.to_msgpack('data/z6_ts_merged_train_full.msgpack')


if __name__ == "__main__":
    O2OEvents.main()
    MerchantFeature.main()
    CouponFeature.main()
    UserFeature.main()
    UserMerchantFeature.main()
    MergedFeature.main()
