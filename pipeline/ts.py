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


try:
    profile
except NameError:
    def profile(f):
        return f


def with_discount(df, df_discount):
    return pd.merge(df, df_discount, how='left', left_on='discount_name', right_index=True)


class ValidateSplit:
    feature_begin = '2016-01-01'
    feature_end = '2016-04-30'
    train_begin = '2016-03-16'
    train_end = '2016-04-30'
    test_begin = '2016-05-16'
    test_end = '2016-05-31'
    test_has_label = True


class TestSplit:
    feature_begin = '2016-01-01'
    feature_end = '2016-06-15'
    train_begin = '2016-03-16'
    train_end = '2016-05-31'
    test_begin = '2016-07-01'
    test_end = '2016-07-31'
    test_has_label = False


def build_date_ranges():
    d = np.arange(366)
    future_end = np.clip(d + 30, 0, 365)
    future_begin = np.clip(d + 1, 0, 365)
    recent_end = np.clip(d - 16, 0, 365)
    recent_begin = np.clip(d - 45, 0, 365)
    history_end = np.clip(d - 46, 0, 365)
    history_begin = np.clip(d - 75, 0, 365)
    longago_end = np.clip(d - 76, 0, 365)
    longago_begin = np.clip(d - 180, 0, 365)
    return list(zip(
        future_end,
        future_begin,
        recent_end,
        recent_begin,
        history_end,
        history_begin,
        longago_end,
        longago_begin,
    ))


DATE_RANGES = build_date_ranges()


class O2OEvents:
    """
    Event types:
        offline_receive_coupon
        offline_buy_with_coupon
        offline_buy_without_coupon
        online_receive_coupon
        online_buy_with_coupon
        online_buy_without_coupon
        online_click

    Event columns:
        date
        date2
        event_type
        user_id
        merchant_id
        coupon_id
        discount_name
        distance
        click_count
        label
    """

    def __init__(self, user_id_index):
        self._data = []
        self.user_id_index = user_id_index

    @staticmethod
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

    def _feed_coupon(self, df, line):
        event_type = self._get_coupon_event_type(df, line)
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

    def to_frame_full(self):
        df = pd.concat(self._data, ignore_index=True).reset_index(drop=True)
        df['event_type'] = df['event_type'].astype('category')
        df['discount_name'] = df['discount_name'].astype('category')
        df.sort_values(['date', 'event_type'], inplace=True)
        return df

    def _label_of(self, df):
        return (
            (df['event_type'] == 'offline_receive_coupon') &
            (df['date2'].notnull()) &
            ((df['date2'].dt.dayofyear - df['date'].dt.dayofyear) <= 15)
        )

    def to_frame(self, split):
        df = self.to_frame_full()
        df1 = df[
            (split.feature_begin <= df['date']) & (df['date'] <= split.feature_end)
        ].copy()
        df1['label'] = self._label_of(df1)
        df2 = df[
            (split.test_begin <= df['date']) & (df['date'] <= split.test_end) &
            (df['event_type'] == 'offline_receive_coupon')
        ].copy()
        df2['label'] = self._label_of(df2)
        df2['date2'] = pd.NaT
        df = pd.concat([df1, df2], ignore_index=True).reset_index(drop=True)
        return df

    @staticmethod
    def build_discount_table(df_events):
        discounts = pd.Series(np.array(df_events['discount_name'].dropna().unique()))
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
        return discount_table

    @staticmethod
    def build_index_of(df, key):
        if isinstance(key, (tuple, list)):
            v = pd.unique(list(df[key].dropna().itertuples(index=False, name=None)))
        else:
            v = np.array(df[key].dropna().unique())
        return v

    @classmethod
    def main(cls, split):
        df_raw_offline = pd.read_msgpack(f'data/z1_raw_offline.msgpack')
        df_raw_test = pd.read_msgpack(f'data/z1_raw_test.msgpack')
        user_id_index = np.unique(np.concatenate([
            df_raw_offline['user_id'].unique(),
            df_raw_test['user_id'].unique()
        ]))
        events = cls(user_id_index)
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
        df = events.to_frame(split)
        df.to_msgpack('data/z6_ts_events.msgpack')

        LOG.info('build_discount_table')
        df_discount = cls.build_discount_table(df)
        df_discount.to_msgpack('data/z6_ts_discount.msgpack')

        df_offline_events = df[
            df['event_type'].isin([
                'offline_receive_coupon',
                'offline_buy_with_coupon',
                'offline_buy_without_coupon',
            ])
        ]
        LOG.info('build_index_of user_id')
        user_id_index = cls.build_index_of(df_offline_events, 'user_id')
        np.save('data/z6_ts_user_id.npy', user_id_index)

        for key in ['merchant_id', 'coupon_id']:
            LOG.info('build_index_of {}', key)
            arr = cls.build_index_of(df_offline_events, key)
            np.save('data/z6_ts_{}.npy'.format(key), arr)

        LOG.info('build_index_of user_id_merchant_id')
        arr = cls.build_index_of(df_offline_events, ['user_id', 'merchant_id'])
        np.save('data/z6_ts_user_id_merchant_id.npy', arr)


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
            'offline_buy_without_coupon',
            'online_buy_without_coupon',
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
        'deltanow',
        'future',
        'today',
        'recent',
        'history',
        'longago',
    ]

    def __init__(self, key, t):
        self.key = key
        self.t = t
        (
            self.future_end,
            self.future_begin,
            self.recent_end,
            self.recent_begin,
            self.history_end,
            self.history_begin,
            self.longago_end,
            self.longago_begin,
        ) = DATE_RANGES[t]

    @property
    def ops(self):
        return {
            'deltanow': self.op_deltanow,
            'future': self.op_future,
            'today': self.op_today,
            'recent': self.op_recent,
            'history': self.op_history,
            'longago': self.op_longago,
        }

    def __repr__(self):
        return '({} key={!r} t={!r})'.format(self.__class__.__name__, self.key, self.t)

    @profile
    def op_deltanow(self, arr):
        return arr['t'] - self.t

    @profile
    def op_future(self, events):
        return events.loc(self.key, self.future_begin, self.future_end)

    @profile
    def op_today(self, events):
        return events.loc(self.key, self.t, self.t)

    @profile
    def op_recent(self, events):
        return events.loc(self.key, self.recent_begin, self.recent_end)

    @profile
    def op_history(self, events):
        return events.loc(self.key, self.history_begin, self.history_end)

    @profile
    def op_longago(self, events):
        return events.loc(self.key, self.longago_begin, self.longago_end)


@profile
def op_unique_coupon(arr):
    return np.unique(arr['coupon_id'])


@profile
def op_unique_hotcoupon(arr):
    vals, cnts = np.unique(arr['coupon_id'], return_counts=True)
    return vals[cnts >= 3]


@profile
def op_unique_user(arr):
    return np.unique(arr['user_id'])


@profile
def op_unique_hotuser(arr):
    vals, cnts = np.unique(arr['user_id'], return_counts=True)
    return vals[cnts >= 2]


@profile
def op_unique_merchant(arr):
    return np.unique(arr['merchant_id'])


@profile
def op_offline_receive_coupon(arr):
    return arr[arr['is_offline_receive_coupon']]


@profile
def op_offline_buy_with_coupon(arr):
    return arr[arr['is_offline_buy_with_coupon']]


@profile
def op_offline_buy_without_coupon(arr):
    return arr[arr['is_offline_buy_without_coupon']]


@profile
def op_offline(arr):
    return arr[arr['is_offline']]


@profile
def op_online_receive_coupon(arr):
    return arr[arr['is_online_receive_coupon']]


@profile
def op_online_buy_with_coupon(arr):
    return arr[arr['is_online_buy_with_coupon']]


@profile
def op_online_buy_without_coupon(arr):
    return arr[arr['is_online_buy_without_coupon']]


@profile
def op_online_click(arr):
    return arr[arr['is_online_click']]


@profile
def op_online(arr):
    return arr[arr['is_online']]


@profile
def op_discount_rate(arr):
    return arr['discount_rate']


@profile
def op_distance(arr):
    return arr['distance']


@profile
def op_timedelta(arr):
    return arr['t'] - arr['t2']


@profile
def op_man200(arr):
    return arr[arr['discount_man'] >= 200]


@profile
def op_count(arr):
    return len(arr)


@profile
def op_min(arr):
    # TODO: FloatingPointError
    return np.min(arr) if len(arr) > 0 else 0


@profile
def op_max(arr):
    return np.max(arr) if len(arr) > 0 else 0


@profile
def op_mean(arr):
    return np.mean(arr) if len(arr) > 0 else 0


@profile
def op_div(a, b):
    if b == 0 or math.isclose(b, 0):
        return 0
    return a / b


@profile
def op_sub(a, b):
    return a - b


CALCULATION_OPS = {
    'unique_coupon': op_unique_coupon,
    'unique_hotcoupon': op_unique_hotcoupon,
    'unique_user': op_unique_user,
    'unique_hotuser': op_unique_hotuser,
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
                steps = []
                name = ''
                for step in tuple(line.split('.')):
                    name += '.' + step
                    steps.append((name, step))
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
            for name, step in steps:
                ops.add(step)
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
            for name, step in steps:
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
        del self.df_events
        del self.df_discount
        events = IndexedEvents(df, key_column=self.key_column)
        result = []
        for key in tqdm.tqdm(self.keys, desc=self.name):
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

    def process_snapshot(self, events, key, t):
        raise NotImplementedError


USER_FEATURE = """
# 最近
# 领券数
recent.offline_receive_coupon.count
# 用券购买数
recent.offline_buy_with_coupon.count
# 无券购买数
recent.offline_buy_without_coupon.count
# 用券购买数/领券数
recent(offline_buy_with_coupon.count / offline_receive_coupon.count)
# 满200领券数
recent.offline_receive_coupon.man200.count
# 满200用券购买数
recent.offline_buy_with_coupon.man200.count
# 满200用券购买数/满200领券数
recent(offline_buy_with_coupon.man200.count / offline_receive_coupon.man200.count)
# 满200用券购买数/用券购买数
recent(offline_buy_with_coupon.man200.count / offline_buy_with_coupon.count)
# 领券.折扣
recent.offline_receive_coupon.discount_rate.mean
recent.offline_receive_coupon.discount_rate.min
recent.offline_receive_coupon.discount_rate.max
# 用券购买.折扣
recent.offline_buy_with_coupon.discount_rate.mean
recent.offline_buy_with_coupon.discount_rate.min
recent.offline_buy_with_coupon.discount_rate.max
# 核销时间
recent.offline_buy_with_coupon.timedelta.min
recent.offline_buy_with_coupon.timedelta.max
recent.offline_buy_with_coupon.timedelta.mean
# 独立商家数量
recent.offline.unique_merchant.count
# 核销独立商家数量
recent.offline_buy_with_coupon.unique_merchant.count
# 核销独立商家数量 / 独立商家数量
recent(offline_buy_with_coupon.unique_merchant.count / offline.unique_merchant.count)
# 独立优惠券数量
recent.offline.unique_coupon.count
# 核销独立优惠券数量
recent.offline_buy_with_coupon.unique_coupon.count
# 核销独立优惠券数量 / 独立优惠券数量
recent(offline_buy_with_coupon.unique_coupon.count / offline.unique_coupon.count)
# 用券购买数 / 核销独立商家数量
recent(offline_buy_with_coupon.count / offline_buy_with_coupon.unique_merchant.count)
# 用券购买.距离
recent.offline_buy_with_coupon.distance.mean
recent.offline_buy_with_coupon.distance.min
recent.offline_buy_with_coupon.distance.max
# 无券购买.距离
recent.offline_buy_without_coupon.distance.mean
recent.offline_buy_without_coupon.distance.min
recent.offline_buy_without_coupon.distance.max

# 很久之前
# 领券数
longago.offline_receive_coupon.count
# 用券购买数
longago.offline_buy_with_coupon.count
# 无券购买数
longago.offline_buy_without_coupon.count
# 用券购买数/领券数
longago(offline_buy_with_coupon.count / offline_receive_coupon.count)
# 满200领券数
longago.offline_receive_coupon.man200.count
# 满200用券购买数
longago.offline_buy_with_coupon.man200.count
# 满200用券购买数/满200领券数
longago(offline_buy_with_coupon.man200.count / offline_receive_coupon.man200.count)
# 满200用券购买数/用券购买数
longago(offline_buy_with_coupon.man200.count / offline_buy_with_coupon.count)
# 领券.折扣
longago.offline_receive_coupon.discount_rate.mean
longago.offline_receive_coupon.discount_rate.min
longago.offline_receive_coupon.discount_rate.max
# 用券购买.折扣
longago.offline_buy_with_coupon.discount_rate.mean
longago.offline_buy_with_coupon.discount_rate.min
longago.offline_buy_with_coupon.discount_rate.max
# 核销时间
longago.offline_buy_with_coupon.timedelta.min
longago.offline_buy_with_coupon.timedelta.max
longago.offline_buy_with_coupon.timedelta.mean
# 独立商家数量
longago.offline.unique_merchant.count
# 核销独立商家数量
longago.offline_buy_with_coupon.unique_merchant.count
# 核销独立商家数量 / 独立商家数量
longago(offline_buy_with_coupon.unique_merchant.count / offline.unique_merchant.count)
# 独立优惠券数量
longago.offline.unique_coupon.count
# 核销独立优惠券数量
longago.offline_buy_with_coupon.unique_coupon.count
# 核销独立优惠券数量 / 独立优惠券数量
longago(offline_buy_with_coupon.unique_coupon.count / offline.unique_coupon.count)
# 用券购买数 / 核销独立商家数量
longago(offline_buy_with_coupon.count / offline_buy_with_coupon.unique_merchant.count)
# 用券购买.距离
longago.offline_buy_with_coupon.distance.mean
longago.offline_buy_with_coupon.distance.min
longago.offline_buy_with_coupon.distance.max
# 无券购买.距离
longago.offline_buy_without_coupon.distance.mean
longago.offline_buy_without_coupon.distance.min
longago.offline_buy_without_coupon.distance.max

# 历史:
# 领券数
history.offline_receive_coupon.count
# 用券购买数
history.offline_buy_with_coupon.count
# 无券购买数
history.offline_buy_without_coupon.count
# 用券购买数/领券数
history(offline_buy_with_coupon.count / offline_receive_coupon.count)
# 满200领券数
history.offline_receive_coupon.man200.count
# 满200用券购买数
history.offline_buy_with_coupon.man200.count
# 满200用券购买数/满200领券数
history(offline_buy_with_coupon.man200.count / offline_receive_coupon.man200.count)
# 满200用券购买数/用券购买数
history(offline_buy_with_coupon.man200.count / offline_buy_with_coupon.count)
# 领券.折扣
history.offline_receive_coupon.discount_rate.mean
history.offline_receive_coupon.discount_rate.min
history.offline_receive_coupon.discount_rate.max
# 用券购买.折扣
history.offline_buy_with_coupon.discount_rate.mean
history.offline_buy_with_coupon.discount_rate.min
history.offline_buy_with_coupon.discount_rate.max
# 核销时间
history.offline_buy_with_coupon.timedelta.min
history.offline_buy_with_coupon.timedelta.max
history.offline_buy_with_coupon.timedelta.mean
# 独立商家数量
history.offline.unique_merchant.count
# 核销独立商家数量
history.offline_buy_with_coupon.unique_merchant.count
# 核销独立商家数量 / 独立商家数量
history(offline_buy_with_coupon.unique_merchant.count / offline.unique_merchant.count)
# 独立优惠券数量
history.offline.unique_coupon.count
# 核销独立优惠券数量
history.offline_buy_with_coupon.unique_coupon.count
# 核销独立优惠券数量 / 独立优惠券数量
history(offline_buy_with_coupon.unique_coupon.count / offline.unique_coupon.count)
# 用券购买数 / 核销独立商家数量
history(offline_buy_with_coupon.count / offline_buy_with_coupon.unique_merchant.count)
# 用券购买.距离
history.offline_buy_with_coupon.distance.mean
history.offline_buy_with_coupon.distance.min
history.offline_buy_with_coupon.distance.max
# 无券购买.距离
history.offline_buy_without_coupon.distance.mean
history.offline_buy_without_coupon.distance.min
history.offline_buy_without_coupon.distance.max

# 今天
# 领券数
today.offline_receive_coupon.count

# 未来
# 领券数
future.offline_receive_coupon.count
# 独立商家数量
future.offline_receive_coupon.unique_merchant.count
# 独立优惠券数量
future.offline_receive_coupon.unique_coupon.count
# 领取时间
future.offline_receive_coupon.deltanow.min

# 线上
# 线上点击次数
history.online_click.count
# 用户线上领取次数
history.online_receive_coupon.count
# 用户线上核销次数
history.online_buy_with_coupon.count
# 用户线上购买次数
history.online_buy_without_coupon.count
# 用户线上优惠券领取率
history(online_receive_coupon.count / online_click.count)
# 用户线上优惠券核销率
history(online_buy_with_coupon.count / online_receive_coupon.count)

# 线上点击次数
recent.online_click.count
# 用户线上领取次数
recent.online_receive_coupon.count
# 用户线上核销次数
recent.online_buy_with_coupon.count
# 用户线上购买次数
recent.online_buy_without_coupon.count
# 用户线上优惠券领取率
recent(online_receive_coupon.count / online_click.count)
# 用户线上优惠券核销率
recent(online_buy_with_coupon.count / online_receive_coupon.count)
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
# 领取时间
future.offline_receive_coupon.deltanow.min
future.offline_receive_coupon.count
today.offline_receive_coupon.count
history.offline_receive_coupon.count
history.offline_buy_with_coupon.count
history.offline_buy_without_coupon.count
history(offline_receive_coupon.count - offline_buy_with_coupon.count)
history(offline_buy_with_coupon.count / offline_receive_coupon.count)

recent.offline_receive_coupon.count
recent.offline_buy_with_coupon.count
recent.offline_buy_without_coupon.count
recent(offline_receive_coupon.count - offline_buy_with_coupon.count)
recent(offline_buy_with_coupon.count / offline_receive_coupon.count)

longago.offline_receive_coupon.count
longago.offline_buy_with_coupon.count
longago.offline_buy_without_coupon.count
longago(offline_receive_coupon.count - offline_buy_with_coupon.count)
longago(offline_buy_with_coupon.count / offline_receive_coupon.count)
"""
UserMerchantFeatureExtractor = FeatureExtractor(USER_MERCHANT_FEATURE)


class UserMerchantFeature(BaseFeature):

    KEY_COLUMN = ['user_id', 'merchant_id']
    COLUMNS = [
        'user_id',
        'merchant_id',
        'date',
        *UserMerchantFeatureExtractor.features,
    ]

    def process_snapshot(self, events, key, t):
        user_id, merchant_id = key
        features = UserMerchantFeatureExtractor.extract(events, key, t)
        return features

    @staticmethod
    def main():
        with TLOG('read data'):
            df = pd.read_msgpack('data/z6_ts_events.msgpack')
            df_discount = pd.read_msgpack('data/z6_ts_discount.msgpack')
            ids = np.load('data/z6_ts_user_id_merchant_id.npy')
        feature = UserMerchantFeature(df, df_discount=df_discount, keys=ids)
        df = feature.process()
        df.to_msgpack('data/z6_ts_feature_user_merchant.msgpack')


COUPON_FEATURE = """
future.offline_receive_coupon.count
today.offline_receive_coupon.count

history.offline_receive_coupon.count
history.offline_receive_coupon.unique_user.count
history.offline_receive_coupon.distance.mean
history.offline_receive_coupon.distance.min
history.offline_receive_coupon.distance.max
history.offline_buy_with_coupon.count
history.offline_buy_with_coupon.unique_user.count
history.offline_buy_with_coupon.timedelta.mean
history.offline_buy_with_coupon.distance.mean
history.offline_buy_with_coupon.distance.min
history.offline_buy_with_coupon.distance.max
history(offline_buy_with_coupon.count / offline_receive_coupon.count)

recent.offline_receive_coupon.count
recent.offline_receive_coupon.unique_user.count
recent.offline_receive_coupon.distance.mean
recent.offline_receive_coupon.distance.min
recent.offline_receive_coupon.distance.max
recent.offline_buy_with_coupon.count
recent.offline_buy_with_coupon.unique_user.count
recent.offline_buy_with_coupon.timedelta.mean
recent.offline_buy_with_coupon.distance.mean
recent.offline_buy_with_coupon.distance.min
recent.offline_buy_with_coupon.distance.max
recent(offline_buy_with_coupon.count / offline_receive_coupon.count)

longago.offline_receive_coupon.count
longago.offline_receive_coupon.unique_user.count
longago.offline_receive_coupon.distance.mean
longago.offline_receive_coupon.distance.min
longago.offline_receive_coupon.distance.max
longago.offline_buy_with_coupon.count
longago.offline_buy_with_coupon.unique_user.count
longago.offline_buy_with_coupon.timedelta.mean
longago.offline_buy_with_coupon.distance.mean
longago.offline_buy_with_coupon.distance.min
longago.offline_buy_with_coupon.distance.max
longago(offline_buy_with_coupon.count / offline_receive_coupon.count)
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


MERCHANT_FEATURE = """
# 最近
# 领券数
recent.offline_receive_coupon.count
# 用券购买数
recent.offline_buy_with_coupon.count
# 无券购买数
recent.offline_buy_without_coupon.count
# 用券购买数/领券数 = 转化率
recent(offline_buy_with_coupon.count / offline_receive_coupon.count)
# 独立用户数
recent.offline.unique_user.count
# 领券独立用户数
recent.offline_receive_coupon.unique_user.count
# 用券购买独立用户数
recent.offline_buy_with_coupon.unique_user.count
# 用券购买独立回头客数
recent.offline_buy_with_coupon.unique_hotuser.count
# 无券购买独立用户数
recent.offline_buy_without_coupon.unique_user.count
# 无券购买独立回头客数
recent.offline_buy_without_coupon.unique_hotuser.count
# 用券购买独立用户数/领券独立用户数 = 用户转化率
recent(offline_buy_with_coupon.unique_user.count / offline_receive_coupon.unique_user.count)
# 用券购买数/用券购买独立用户数 = 人均用券购买数
recent(offline_buy_with_coupon.count / offline_buy_with_coupon.unique_user.count)
# 领券.折扣
recent.offline_receive_coupon.discount_rate.min
recent.offline_receive_coupon.discount_rate.max
recent.offline_receive_coupon.discount_rate.mean
# 用券购买.折扣
recent.offline_buy_with_coupon.discount_rate.min
recent.offline_buy_with_coupon.discount_rate.max
recent.offline_buy_with_coupon.discount_rate.mean
# 用券购买.距离
recent.offline_buy_with_coupon.distance.min
recent.offline_buy_with_coupon.distance.max
recent.offline_buy_with_coupon.distance.mean
# 无券购买.距离
recent.offline_buy_without_coupon.distance.min
recent.offline_buy_without_coupon.distance.max
recent.offline_buy_without_coupon.distance.mean
# 用券购买.时间间隔
recent.offline_buy_with_coupon.timedelta.min
recent.offline_buy_with_coupon.timedelta.max
recent.offline_buy_with_coupon.timedelta.mean
# 优惠券种类数目
recent.offline.unique_coupon.count
# 核销>=3次优惠券种数
recent.offline.unique_hotcoupon.count

# 很久以前
# 领券数
longago.offline_receive_coupon.count
# 用券购买数
longago.offline_buy_with_coupon.count
# 无券购买数
longago.offline_buy_without_coupon.count
# 用券购买数/领券数 = 转化率
longago(offline_buy_with_coupon.count / offline_receive_coupon.count)
# 独立用户数
longago.offline.unique_user.count
# 领券独立用户数
longago.offline_receive_coupon.unique_user.count
# 用券购买独立用户数
longago.offline_buy_with_coupon.unique_user.count
# 用券购买独立回头客数
longago.offline_buy_with_coupon.unique_hotuser.count
# 无券购买独立用户数
longago.offline_buy_without_coupon.unique_user.count
# 无券购买独立回头客数
longago.offline_buy_without_coupon.unique_hotuser.count
# 用券购买独立用户数/领券独立用户数 = 用户转化率
longago(offline_buy_with_coupon.unique_user.count / offline_receive_coupon.unique_user.count)
# 用券购买数/用券购买独立用户数 = 人均用券购买数
longago(offline_buy_with_coupon.count / offline_buy_with_coupon.unique_user.count)
# 领券.折扣
longago.offline_receive_coupon.discount_rate.min
longago.offline_receive_coupon.discount_rate.max
longago.offline_receive_coupon.discount_rate.mean
# 用券购买.折扣
longago.offline_buy_with_coupon.discount_rate.min
longago.offline_buy_with_coupon.discount_rate.max
longago.offline_buy_with_coupon.discount_rate.mean
# 用券购买.距离
longago.offline_buy_with_coupon.distance.min
longago.offline_buy_with_coupon.distance.max
longago.offline_buy_with_coupon.distance.mean
# 无券购买.距离
longago.offline_buy_without_coupon.distance.min
longago.offline_buy_without_coupon.distance.max
longago.offline_buy_without_coupon.distance.mean
# 用券购买.时间间隔
longago.offline_buy_with_coupon.timedelta.min
longago.offline_buy_with_coupon.timedelta.max
longago.offline_buy_with_coupon.timedelta.mean
# 优惠券种类数目
longago.offline.unique_coupon.count
# 核销>=3次优惠券种数
longago.offline.unique_hotcoupon.count

# 历史:
# 领券数
history.offline_receive_coupon.count
# 用券购买数
history.offline_buy_with_coupon.count
# 无券购买数
history.offline_buy_without_coupon.count
# 用券购买数/领券数 = 转化率
history(offline_buy_with_coupon.count / offline_receive_coupon.count)
# 独立用户数
history.offline.unique_user.count
# 领券独立用户数
history.offline_receive_coupon.unique_user.count
# 用券购买独立用户数
history.offline_buy_with_coupon.unique_user.count
# 用券购买独立回头客数
history.offline_buy_with_coupon.unique_hotuser.count
# 无券购买独立用户数
history.offline_buy_without_coupon.unique_user.count
# 无券购买独立回头客数
history.offline_buy_without_coupon.unique_hotuser.count
# 用券购买独立用户数/领券独立用户数 = 用户转化率
history(offline_buy_with_coupon.unique_user.count / offline_receive_coupon.unique_user.count)
# 用券购买数/用券购买独立用户数 = 人均用券购买数
history(offline_buy_with_coupon.count / offline_buy_with_coupon.unique_user.count)
# 领券.折扣
history.offline_receive_coupon.discount_rate.min
history.offline_receive_coupon.discount_rate.max
history.offline_receive_coupon.discount_rate.mean
# 用券购买.折扣
history.offline_buy_with_coupon.discount_rate.min
history.offline_buy_with_coupon.discount_rate.max
history.offline_buy_with_coupon.discount_rate.mean
# 用券购买.距离
history.offline_buy_with_coupon.distance.min
history.offline_buy_with_coupon.distance.max
history.offline_buy_with_coupon.distance.mean
# 无券购买.距离
history.offline_buy_without_coupon.distance.min
history.offline_buy_without_coupon.distance.max
history.offline_buy_without_coupon.distance.mean
# 用券购买.时间间隔
history.offline_buy_with_coupon.timedelta.min
history.offline_buy_with_coupon.timedelta.max
history.offline_buy_with_coupon.timedelta.mean
# 优惠券种类数目
history.offline.unique_coupon.count
# 核销>=3次优惠券种数
history.offline.unique_hotcoupon.count

# 今天:
# 领券数
today.offline_receive_coupon.count
# 领券独立用户数
today.offline_receive_coupon.unique_user.count
# 优惠券种类数目
today.offline_receive_coupon.unique_coupon.count

# 未来:
# 领券数
future.offline_receive_coupon.count
# 领券独立用户数
future.offline_receive_coupon.unique_user.count
# 优惠券种类数目
future.offline_receive_coupon.unique_coupon.count
"""
MerchantFeatureExtractor = FeatureExtractor(MERCHANT_FEATURE)


class MerchantFeature(BaseFeature):

    KEY_COLUMN = 'merchant_id'
    COLUMNS = [
        'merchant_id',
        'date',
        *MerchantFeatureExtractor.features,
    ]

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


def with_date_feature(df):
    t = pd.Timestamp('2016-01-01') + pd.to_timedelta(df['date'], unit='d')
    df['date_dayofweek'] = t.dt.dayofweek
    df['date_dayofmonth'] = t.dt.day
    df['date'] = t


class MergedFeature:

    @staticmethod
    def prefix_columns(df, prefix):
        cols = []
        sp_cols = {'date', 'user_id', 'merchant_id', 'coupon_id', 'label'}
        for x in df.columns:
            x = str(x)
            if x not in sp_cols:
                x = prefix + x
            cols.append(x)
        df = df.copy()
        df.columns = cols
        return df

    @classmethod
    def merge_feature(cls, df, df_user, df_merchant, df_coupon, df_user_merchant):
        df = df.copy()
        df['date'] = df['date'].dt.dayofyear - 1
        df_user = cls.prefix_columns(df_user, 'user:')
        df_merchant = cls.prefix_columns(df_merchant, 'merchant:')
        df_coupon = cls.prefix_columns(df_coupon, 'coupon:')
        df_user_merchant = cls.prefix_columns(df_user_merchant, 'user_merchant:')
        df = pd.merge(df, df_user, how='left',
                      left_on=['user_id', 'date'],
                      right_on=['user_id', 'date'])

        df = pd.merge(df, df_merchant, how='left',
                      left_on=['merchant_id', 'date'],
                      right_on=['merchant_id', 'date'])

        df = pd.merge(df, df_coupon, how='left',
                      left_on=['coupon_id', 'date'],
                      right_on=['coupon_id', 'date'])

        df = pd.merge(df, df_user_merchant, how='left',
                      left_on=['user_id', 'merchant_id', 'date'],
                      right_on=['user_id', 'merchant_id', 'date'])

        with_date_feature(df)
        df = df.reset_index(drop=True)
        df_full = df.copy()

        df = df.drop(['user_id', 'merchant_id', 'coupon_id', 'discount_name'], axis=1)
        df.fillna(0, inplace=True)

        return df, df_full

    @staticmethod
    def main(split):
        LOG.info('read features')
        df_user = pd.read_msgpack('data/z6_ts_feature_user.msgpack')
        df_merchant = pd.read_msgpack('data/z6_ts_feature_merchant.msgpack')
        df_coupon = pd.read_msgpack('data/z6_ts_feature_coupon.msgpack')
        df_user_merchant = pd.read_msgpack('data/z6_ts_feature_user_merchant.msgpack')
        df_coupon = pd.read_msgpack('data/z6_ts_feature_coupon.msgpack')
        df_discount = pd.read_msgpack('data/z6_ts_discount.msgpack')
        df_events = pd.read_msgpack('data/z6_ts_events.msgpack')

        LOG.info('split train/test')
        common_cols = [
            'user_id', 'merchant_id', 'coupon_id',
            'date', 'distance', 'discount_name',
        ]
        mask = df_events['event_type'] == 'offline_receive_coupon'
        df_train = df_events.loc[
            mask & (df_events['date'] >= split.train_begin) &
            (df_events['date'] <= split.train_end),
            common_cols + ['label']
        ].copy()
        if split.test_has_label:
            test_cols = common_cols + ['label']
        else:
            test_cols = common_cols
        df_test = df_events.loc[
            mask & (df_events['date'] >= split.test_begin) &
            (df_events['date'] <= split.test_end),
            test_cols
        ].copy()

        LOG.info('merge for test')
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
    split = TestSplit
    O2OEvents.main(split=split)
    MerchantFeature.main()
    CouponFeature.main()
    UserFeature.main()
    UserMerchantFeature.main()
    MergedFeature.main(split=split)
