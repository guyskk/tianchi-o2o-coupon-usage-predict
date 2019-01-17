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
import math
import time
from collections import OrderedDict
from contextlib import contextmanager

import numpy as np
import pandas as pd
import tqdm
from loguru import logger as LOG

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


def with_discount(df, df_discount):
    return pd.merge(df, df_discount, how='left', left_on='discount_name', right_index=True)


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


try:
    profile
except NameError:
    def profile(f):
        return f


class IndexedEvents:
    def __init__(self, df_events, key_column):
        self.key_column = key_column
        self.is_multikey = not isinstance(key_column, str)

        with TLOG('dropna'):
            if self.is_multikey:
                df_events = df_events.loc[df_events[key_column].dropna().index]
            else:
                df_events = df_events[df_events[key_column].notnull()]

        with TLOG('_wide_df'):
            df_events = self._wide_df(df_events.copy())

        with TLOG('sort_values'):
            cols = ['t', 'is_offline', 'is_coupon', 'is_buy']
            if self.is_multikey:
                cols = self.key_column + cols
            else:
                cols = [self.key_column] + cols
            df_events = df_events.sort_values(cols)
        self.df_events = df_events

        dtype = self._get_np_dtype(df_events)
        self._data = np.empty((len(df_events),), dtype=np.dtype(dtype, align=True))
        with TLOG('convert to numpy'):
            for name in df_events.columns:
                self._data[name] = df_events[name].values

        self._index = {}
        with TLOG('_build_index'):
            if self.is_multikey:
                values = self._multikey_values(df_events)
            else:
                values = df_events[[self.key_column, 't']].values
            self._build_index(values)

    def _multikey_values(self, df):
        values = df[[*self.key_column, 't']].values
        for *key, t in values:
            yield tuple(key), t

    def _wide_df(self, df):
        df['is_offline'] = df['event_type'].isin((
            'offline_receive_coupon',
            'offline_buy_with_coupon',
            'offline_buy_without_coupon',
        ))
        df['is_coupon'] = df['event_type'].isin((
            'offline_receive_coupon',
            'online_receive_coupon',
            'offline_buy_with_coupon',
            'online_buy_with_coupon',
        ))
        df['is_buy'] = df['event_type'].isin((
            'offline_buy_with_coupon',
            'online_buy_with_coupon',
        ))
        df['is_offline_receive_coupon'] = df['is_offline'] & df['is_coupon'] & (~df['is_buy'])
        df['is_offline_buy_with_coupon'] = df['is_offline'] & df['is_coupon'] & (df['is_buy'])
        df['is_offline_buy_without_coupon'] = df['is_offline'] & (~df['is_coupon']) & df['is_buy']
        df['is_online_receive_coupon'] = (~df['is_offline']) & df['is_coupon'] & (~df['is_buy'])
        df['is_online_buy_with_coupon'] = (~df['is_offline']) & df['is_coupon'] & (df['is_buy'])
        df['is_online_buy_without_coupon'] = (~df['is_offline']) & (~df['is_coupon']) & df['is_buy']
        df['is_online_click'] = (~df['is_offline']) & (~df['is_coupon']) & (~df['is_buy'])
        df['t'] = df['date'].dt.dayofyear - 1
        df['t2'] = df['date2'].dt.dayofyear - 1
        return df

    def _get_np_dtype(self, df):
        np_dtype = []
        for k, v in df.dtypes.items():
            if v == np.object:
                v = np.str
            else:
                try:
                    v = np.dtype(v)
                except TypeError:
                    v = np.str
            np_dtype.append((k, v))
        return np_dtype

    @profile
    def _build_index(self, values):
        prev_key = None
        prev_t = 0
        for i, (key, t) in enumerate(values):
            if key == prev_key:
                idx = self._index[key]
                if t == prev_t:
                    # prev merchant, prev date
                    pass
                else:
                    # prev merchant, new date
                    idx[prev_t, 1] = i - 1
                    idx[prev_t+1:t, 0] = i - 1
                    idx[t, 0] = i
            else:
                # new merchant, new date
                idx = np.full((366, 2), -1)
                idx[:t, 0] = i - 1
                idx[t, 0] = i
                self._index[key] = idx
                if prev_key is not None:
                    prev_idx = self._index[prev_key]
                    prev_idx[prev_t, 1] = i - 1
                    prev_idx[prev_t+1:, 0] = i - 1
            prev_key = key
            prev_t = t
        if prev_key is not None:
            prev_idx = self._index[prev_key]
            prev_idx[prev_t, 1] = i - 1
            prev_idx[prev_t+1:, 0] = i - 1

    def loc(self, key, t1=None, t2=None):
        idx = self._index[key]
        if t1 is None:
            t1 = 0
        if t2 is None:
            t2 = 365
        i1, j1 = idx[t1]
        i2, j2 = idx[t2]
        if j1 == -1:
            i = i1 + 1
        else:
            i = i1
        if j2 == -1:
            j = i2 + 1
        else:
            j = j2 + 1
        return self._data[i:j]


class EventsSnapshot:

    OP_NAMES = [
        'history',
        'future',
        'recent',
        'today',
        'yesterday',
    ]

    def __init__(self, key, t):
        self.key = key
        self.t = t
        (
            self.yesterday,
            self.future_begin,
            self.future_end,
            self.recent_end,
            self.recent_begin,
            self.history_end,
        ) = DATE_RANGES[t]

    @property
    def ops(self):
        return {
            'history': self.op_history,
            'future': self.op_future,
            'recent': self.op_recent,
            'today': self.op_today,
            'yesterday': self.op_yesterday,

        }

    def __repr__(self):
        return '({} key={!r} t={!r})'.format(self.__class__.__name__, self.key, self.t)

    def op_history(self, events):
        return events.loc(self.key, None, self.history_end)

    def op_future(self, events):
        return events.loc(self.key, self.future_begin, self.future_end)

    def op_recent(self, events):
        return events.loc(self.key, self.recent_begin, self.recent_end)

    def op_today(self, events):
        return events.loc(self.key, self.t, self.t)

    def op_yesterday(self, events):
        return events.loc(self.key, self.yesterday, self.yesterday)


def op_unique_coupon(arr):
    return np.unique(arr['coupon_id'])


def op_unique_user(arr):
    return np.unique(arr['user_id'])


def op_unique_merchant(arr):
    return np.unique(arr['user_id'])


def op_offline_receive_coupon(arr):
    return arr[arr['is_offline_receive_coupon']]


def op_offline_buy_with_coupon(arr):
    return arr[arr['is_offline_buy_with_coupon']]


def op_offline_buy_without_coupon(arr):
    return arr[arr['is_offline_buy_without_coupon']]


def op_offline(arr):
    return arr[arr['is_offline']]


def op_online_receive_coupon(arr):
    return arr[arr['is_online_receive_coupon']]


def op_online_buy_with_coupon(arr):
    return arr[arr['is_online_buy_with_coupon']]


def op_online_buy_without_coupon(arr):
    return arr[arr['is_online_buy_without_coupon']]


def op_online_click(arr):
    return arr[arr['is_online_click']]


def op_online(arr):
    return arr[arr['is_online']]


def op_discount_rate(arr):
    return arr['discount_rate']


def op_distance(arr):
    return arr['distance']


def op_timedelta(arr):
    return arr['t'] - arr['t2']


def op_man200(arr):
    return arr[arr['discount_man'] >= 200]


def op_count(arr):
    return len(arr)


def op_min(arr):
    # TODO: FloatingPointError
    return np.min(arr) if len(arr) > 0 else 0


def op_max(arr):
    return np.max(arr) if len(arr) > 0 else 0


def op_mean(arr):
    return np.mean(arr) if len(arr) > 0 else 0


def op_div(a, b):
    if b == 0 or math.isclose(b, 0):
        return 0
    return a / b


def op_sub(a, b):
    return a - b


CALCULATION_OPS = {
    'unique_coupon': op_unique_coupon,
    'unique_user': op_unique_user,
    'unique_merchant': op_unique_merchant,
    'offline_receive_coupon': op_offline_receive_coupon,
    'offline_buy_with_coupon': op_offline_buy_with_coupon,
    'offline_buy_without_coupon': op_offline_buy_without_coupon,
    'offline': op_offline,
    'online_receive_coupon': op_online_receive_coupon,
    'online_buy_with_coupon': op_online_buy_with_coupon,
    'online_buy_without_coupon': op_online_buy_without_coupon,
    'online_click': op_online_click,
    'online': op_online,
    'discount_rate': op_discount_rate,
    'distance': op_distance,
    'timedelta': op_timedelta,
    'man200': op_man200,
    'count': op_count,
    'min': op_min,
    'max': op_max,
    'mean': op_mean,
}

COMBINATION_OPS = {
    'sub': op_sub,
    'div': op_div,
}

OP_NAMES = [
    *EventsSnapshot.OP_NAMES,
    *CALCULATION_OPS.keys(),
    *COMBINATION_OPS.keys(),
]


def build_date_ranges():
    d = np.arange(366)
    yesterday = np.clip(d - 1, 0, 365)
    future_begin = np.clip(d + 1, 0, 365)
    future_end = np.clip(d + 15, 0, 365)
    recent_end = np.clip(d - 15, 0, 365)
    recent_begin = np.clip(d - 74, 0, 365)
    history_end = np.clip(d - 75, 0, 365)
    return list(zip(
        yesterday,
        future_begin,
        future_end,
        recent_end,
        recent_begin,
        history_end,
    ))


DATE_RANGES = build_date_ranges()


class FeatureExtractor:

    def __init__(self, definition):
        self.calculation = []
        self.combination = []
        for line in definition.strip().splitlines():
            line = line.split('#', 1)[0].strip().replace(' ', '')
            if not line:
                continue
            feature = line
            if '(' in line:
                prefix, line = line.split('(', 1)
                line = line[:-1]
                if '-' in line:
                    op = 'sub'
                    left, right = line.split('-', 1)
                else:
                    assert '/' in line, f'invalid syntax {line}'
                    op = 'div'
                    left, right = line.split('/', 1)
                left = '.{}.{}'.format(prefix, left)
                right = '.{}.{}'.format(prefix, right)
                self.combination.append((feature, op, left, right))
            else:
                steps = tuple(line.split('.'))
                self.calculation.append((feature, steps))
        self.features = []
        self.features.extend(x[0] for x in self.calculation)
        self.features.extend(x[0] for x in self.combination)
        self.calculation = [x[1] for x in self.calculation]
        self.combination = [x[1:] for x in self.combination]
        self.check_missing_ops()

    def check_missing_ops(self):
        ops = set()
        for steps in self.calculation:
            ops.update(steps)
        ops.update(x[0] for x in self.combination)
        missing = list(sorted(ops - set(OP_NAMES)))
        if missing:
            raise ValueError('missing ops: {}'.format(missing))

    @profile
    def extract(self, events, key, t):
        snapshot = EventsSnapshot(key, t)
        calculation_ops = {
            **CALCULATION_OPS,
            **snapshot.ops,
        }
        combination_ops = COMBINATION_OPS
        ret = []
        cache = {}
        for steps in self.calculation:
            arr = events
            name = ''
            for step in steps:
                name += '.' + step
                if name in cache:
                    arr = cache[name]
                else:
                    arr = calculation_ops[step](arr)
                    cache[name] = arr
            ret.append(arr)
        for op, left, right in self.combination:
            v = combination_ops[op](cache[left], cache[right])
            ret.append(v)
        return ret


class BaseFeature:

    def __init__(self, df_events, keys, df_discount):
        self.name = self.__class__.__name__
        self.columns = self.__class__.COLUMNS
        self.key_column = self.__class__.KEY_COLUMN
        self.is_multikey = not isinstance(self.key_column, str)
        self.df_events = df_events
        if self.is_multikey:
            keys = [tuple(x) for x in keys]
        self.keys = keys
        self.df_discount = df_discount

    @profile
    def process(self):
        df = with_discount(self.df_events, self.df_discount)
        events = IndexedEvents(df, key_column=self.key_column)
        result = []
        for key in tqdm.tqdm(self.keys):
            sub_events = events.loc(key)
            idx_date = np.unique(sub_events['t'])
            for t in idx_date:
                row = self.process_snapshot(events, key, t)
                if self.is_multikey:
                    result.append([*key, t, *row])
                else:
                    result.append([key, t, *row])
        df_result = pd.DataFrame.from_records(result, columns=self.COLUMNS)
        return df_result

    def process_snapshot(self, key, snapshot):
        raise NotImplementedError


USER_FEATURE = """
recent.offline_receive_coupon.count  # 用户领取优惠券次数
recent.offline_buy_with_coupon.count  # 用户获得优惠券并核销次数
recent.offline_buy_without_coupon.count  # 无券购买次数
# 用户核销优惠券的平均/最低/最高消费折率
recent.offline_buy_with_coupon.discount_rate.mean
recent.offline_buy_with_coupon.discount_rate.min
recent.offline_buy_with_coupon.discount_rate.max
recent.offline_buy_with_coupon.man200.count  # 用户满200~500减的优惠券
recent.offline_receive_coupon.man200.count  # 用户满200~500减的优惠券
recent.offline.unique_merchant.count  # 所有不同商家数量
recent.offline.unique_coupon.count  # 所有不同优惠券数量
recent.offline_buy_with_coupon.unique_merchant.count  # 用户核销过优惠券的不同商家数量
recent.offline_buy_with_coupon.unique_coupon.count  # 用户核销过的不同优惠券数量
# 用户核销优惠券中的平均/最大/最小用户-商家距离
recent.offline_buy_with_coupon.distance.mean
recent.offline_buy_with_coupon.distance.min
recent.offline_buy_with_coupon.distance.max
# 用户核销过优惠券的不同商家数量占所有不同商家的比重
recent(offline_buy_with_coupon.unique_merchant.count / offline.unique_merchant.count)
# 用户核销过优惠券的不同优惠券数量占所有不同优惠券的比重
recent(offline_buy_with_coupon.unique_coupon.count / offline.unique_coupon.count)
recent(offline_receive_coupon.count - offline_buy_with_coupon.count)  # 用户获得优惠券但没有消费的次数
recent(offline_buy_with_coupon.count / offline_receive_coupon.count)  # 用户领取优惠券后进行核销率
# 用户平均核销每个商家多少张优惠券
recent(offline_buy_with_coupon.count / offline_buy_with_coupon.unique_merchant.count)
# 用户满200~500减的优惠券核销率
recent(offline_buy_with_coupon.man200.count / offline_receive_coupon.man200.count)
recent.online_click.count  # 用户线上点击次数
recent.online_receive_coupon.count  # 用户线上领取次数
recent.online_buy_with_coupon.count  # 用户线上核销次数
recent.online_buy_without_coupon.count  # 用户线上购买次数
recent(online_receive_coupon.count / online_click.count)  # 用户线上优惠券领取率
recent(online_buy_with_coupon.count / online_receive_coupon.count)  # 用户线上优惠券核销率
"""
UserFeatureExtractor = FeatureExtractor(USER_FEATURE)


class UserFeature(BaseFeature):

    KEY_COLUMN = 'user_id'
    COLUMNS = [
        'user_id',
        'date',
        *UserFeatureExtractor.features,
    ]

    def process_snapshot(self, events, key, t):
        return UserFeatureExtractor.extract(events, key, t)

    @staticmethod
    def main():
        with TLOG('read data'):
            df = pd.read_msgpack('data/z6_ts_events.msgpack')
            df_discount = pd.read_msgpack('data/z6_ts_discount.msgpack')
            user_ids = np.load('data/z6_ts_user_id.npy')
        user_feature = UserFeature(df, df_discount=df_discount, keys=user_ids)
        df = user_feature.process()
        df.to_msgpack('data/z6_ts_feature_user.msgpack')


USER_MERCHANT_FEATURE = """
recent.offline_receive_coupon.count
recent.offline_buy_with_coupon.count
recent.offline_buy_without_coupon.count
recent(offline_receive_coupon.count - offline_buy_with_coupon.count)
recent(offline_buy_with_coupon.count / offline_receive_coupon.count)
"""
UserMerchantFeatureExtractor = FeatureExtractor(USER_MERCHANT_FEATURE)


class UserMerchantFeature(BaseFeature):

    KEY_COLUMN = ['user_id', 'merchant_id']
    COLUMNS = [
        'user_id',
        'merchant_id',
        'date',
        *UserMerchantFeatureExtractor.features,
        'recent_offline_receive_share_user',
        'recent_offline_buy_share_user',
        'recent_offline_receive_share_merchant',
        'recent_offline_buy_share_merchant',
    ]

    def __init__(self, *args, df_feature_user, df_feature_merchant, **kwargs):
        super().__init__(*args, **kwargs)
        self.df_feature_user = df_feature_user.set_index(['user_id', 'date'])
        self.df_feature_merchant = df_feature_merchant.set_index(['merchant_id', 'date'])

    def process_snapshot(self, events, key, t):
        user_id, merchant_id = key
        features = UserMerchantFeatureExtractor.extract(events, key, t)
        receive_coupon_count = features[0]
        buy_with_coupon_count = features[1]

        # 用户对商家的不核销次数占用户总的不核销次数的比重
        try:
            user_feature = self.df_feature_user.loc[user_id, t]
        except KeyError as ex:
            LOG.exception(ex)
            user_receive_count = 0
            user_buy_with_count = 0
        else:
            user_receive_count = user_feature['recent.offline_receive_coupon.count']
            user_buy_with_count = user_feature['recent.offline_buy_with_coupon.count']
        recent_offline_receive_share_user = op_div(receive_coupon_count, user_receive_count)
        recent_offline_buy_share_user = op_div(buy_with_coupon_count, user_buy_with_count)

        # 用户对每个商家的不核销次数占商家总的不核销次数的比重
        try:
            merchant_feature = self.df_feature_merchant.loc[merchant_id, t]
        except KeyError as ex:
            LOG.exception(ex)
            merchant_receive_count = 0
            merchant_buy_with_count = 0
        else:
            merchant_receive_count = merchant_feature['recent.offline_receive_coupon.count']
            merchant_buy_with_count = merchant_feature['recent.offline_buy_with_coupon.count']
        recent_offline_receive_share_merchant = op_div(
            receive_coupon_count, merchant_receive_count)
        recent_offline_buy_share_merchant = op_div(buy_with_coupon_count, merchant_buy_with_count)

        features.extend([
            recent_offline_receive_share_user,
            recent_offline_buy_share_user,
            recent_offline_receive_share_merchant,
            recent_offline_buy_share_merchant,
        ])

        return features

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


COUPON_FEATURE = """
yesterday.offline_receive_coupon.count
recent.offline_receive_coupon.count
recent.offline_buy_with_coupon.count
recent.offline_buy_with_coupon.unique_user.count
recent.offline_buy_with_coupon.timedelta.mean
recent.offline_buy_with_coupon.distance.mean
recent.offline_buy_with_coupon.distance.min
recent.offline_buy_with_coupon.distance.max
recent(offline_buy_with_coupon.count / offline_receive_coupon.count)
"""
CouponFeatureExtractor = FeatureExtractor(COUPON_FEATURE)


class CouponFeature(BaseFeature):

    KEY_COLUMN = 'coupon_id'
    COLUMNS = [
        'coupon_id',
        'date',
        *CouponFeatureExtractor.features,
    ]

    def process_snapshot(self, events, key, t):
        return CouponFeatureExtractor.extract(events, key, t)

    @staticmethod
    def main():
        with TLOG('read data'):
            df = pd.read_msgpack('data/z6_ts_events.msgpack')
            df_discount = pd.read_msgpack('data/z6_ts_discount.msgpack')
            coupon_ids = np.load('data/z6_ts_coupon_id.npy')
        coupon_feature = CouponFeature(df, df_discount=df_discount, keys=coupon_ids)
        df = coupon_feature.process()
        df.to_msgpack('data/z6_ts_feature_coupon.msgpack')


def with_date_feature(df):
    df['date_dayofweek'] = df['date'].dt.dayofweek
    df['date_dayofmonth'] = df['date'].dt.day


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

        with_date_feature(df)
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


# class LeakFeature(BaseFeature):
#     def process(self, user_id, df):
#         # 今天领了多少张券
#         # 今天领了多少张同商家的券
#         # 今天领了多少张相同的券
#         # 未来15天领了多少张券，及最短时间间隔
#         # 未来15天领了多少张同商家的券，及最短时间间隔
#         # 未来15天领了多少张相同的券，及最短时间间隔
#         # 未来15天领了多少个不同商家
#         # 未来15天领了多少种券
#         # 未来15天商家被领取的优惠券数目
#         # 未来15天商家被领取的相同优惠券数目
#         # 未来15天商家被多少不同用户领取的数目
#         # 未来15天商家发行的所有优惠券种类数目
#         coupon_count = len(df)


MERCHANT_FEATURE = """
yesterday.offline_receive_coupon.count
yesterday.offline_buy_without_coupon.count
recent.unique_coupon.count  # 不同优惠券数量
recent.offline_receive_coupon.count  # 商家优惠券被领取次数
recent.unique_user.count  # 核销商家优惠券的不同用户数量
recent.offline_receive_coupon.unique_user.count
recent.offline_buy_with_coupon.count  # 商家优惠券被领取后核销次数
recent.offline_buy_with_coupon.unique_user.count  # 核销商家优惠券的不同用户数量
recent.offline_buy_with_coupon.unique_coupon.count
recent.offline_buy_with_coupon.discount_rate.min  # 商家优惠券核销的最小消费折率
recent.offline_buy_with_coupon.discount_rate.max  # 商家优惠券核销的最大消费折率
recent.offline_buy_with_coupon.discount_rate.mean  # 商家优惠券核销的平均消费折率
recent.offline_buy_with_coupon.distance.min  # 商家优惠券核销的最小距离
recent.offline_buy_with_coupon.distance.max  # 商家优惠券核销的最大距离
recent.offline_buy_with_coupon.distance.mean  # 商家优惠券核销的平均距离
recent.offline_buy_with_coupon.timedelta.mean  # # 商家被核销优惠券的平均时间率
recent(offline_receive_coupon.count - offline_buy_with_coupon.count)  # 商家优惠券被领取后不核销次数
recent(offline_buy_with_coupon.count / offline_receive_coupon.count)  # 商家优惠券被领取后核销率
recent(offline_buy_with_coupon.unique_user.count / offline_receive_coupon.unique_user.count)  # 不同用户数量占领取不同的用户比重
recent(offline_buy_with_coupon.unique_coupon.count / unique_coupon.count)  # 商家被核销过的不同优惠券数量占所有领取过的不同优惠券数量的比重
"""
MerchantFeatureExtractor = FeatureExtractor(MERCHANT_FEATURE)


class MerchantFeature(BaseFeature):

    KEY_COLUMN = 'merchant_id'
    COLUMNS = [
        'merchant_id',
        'date',
        *MerchantFeatureExtractor.features,
    ]

    @profile
    def process_snapshot(self, events, key, t):
        return MerchantFeatureExtractor.extract(events, key, t)

    @staticmethod
    def main():
        with TLOG('read data'):
            df = pd.read_msgpack('data/z6_ts_events.msgpack')
            df_discount = pd.read_msgpack('data/z6_ts_discount.msgpack')
            merchant_ids = np.load('data/z6_ts_merchant_id.npy')
        merchant_feature = MerchantFeature(df, df_discount=df_discount, keys=merchant_ids)
        df = merchant_feature.process()
        df.to_msgpack('data/z6_ts_feature_merchant.msgpack')


if __name__ == "__main__":
    # O2OEvents.main()
    MerchantFeature.main()
    CouponFeature.main()
    UserFeature.main()
    UserMerchantFeature.main()
    # MergedFeature.main()
    # with TLOG('read events'):
    #     df_events = pd.read_msgpack('data/z6_ts_events.msgpack')
    # ti = IndexedEvents(df_events)
    # t1 = pd.Timestamp('2016-04-20').dayofyear
    # t2 = pd.Timestamp('2016-04-22').dayofyear
    # print(ti.loc(1159, t1, t2))
