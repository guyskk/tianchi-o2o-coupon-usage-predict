from collections import OrderedDict

import numpy as np
import pandas as pd
import tqdm
from loguru import logger as LOG

from z1_raw import apply_discount_rate
from util import TLOG, profile, get_config
from feature_definition import FEATURES, TestSplit, ValidateSplit
from events_snapshot import EventsSnapshot, CALCULATION_OPS, COMBINATION_OPS, OP_NAMES


def with_discount(df, df_discount):
    return pd.merge(df, df_discount, how='left', left_on='discount_name', right_index=True)


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
        df.to_msgpack(f'data/z6_ts_{split.name}_events.msgpack')

        LOG.info('build_discount_table')
        df_discount = cls.build_discount_table(df)
        df_discount.to_msgpack(f'data/z6_ts_{split.name}_discount.msgpack')

        df_offline_events = df[
            df['event_type'].isin([
                'offline_receive_coupon',
                'offline_buy_with_coupon',
                'offline_buy_without_coupon',
            ])
        ]
        LOG.info('build_index_of user_id')
        user_id_index = cls.build_index_of(df_offline_events, 'user_id')
        np.save(f'data/z6_ts_{split.name}_user_id.npy', user_id_index)

        for key in ['merchant_id', 'coupon_id']:
            LOG.info('build_index_of {}', key)
            arr = cls.build_index_of(df_offline_events, key)
            np.save('data/z6_ts_{}_{}.npy'.format(split.name, key), arr)

        LOG.info('build_index_of user_id_merchant_id')
        arr = cls.build_index_of(df_offline_events, ['user_id', 'merchant_id'])
        np.save(f'data/z6_ts_{split.name}_user_id_merchant_id.npy', arr)

        LOG.info('build_index_of user_id_coupon_id')
        arr = cls.build_index_of(df_offline_events, ['user_id', 'coupon_id'])
        np.save(f'data/z6_ts_{split.name}_user_id_coupon_id.npy', arr)


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


class FeatureProcessor:

    def __init__(self, feature, split):
        self.feature = feature
        self.split = split
        self.name = feature.name
        self.is_multikey = not isinstance(feature.key_column, str)
        self.extractor = FeatureExtractor(feature.definition)
        if self.is_multikey:
            self.key_column = list(feature.key_column)
            self.columns = list(feature.key_column)
        else:
            self.key_column = feature.key_column
            self.columns = [feature.key_column]
        self.columns.append('date')
        self.columns.extend(self.extractor.features)

    def read_events(self):
        split = self.split.name
        with TLOG('read events'):
            df_events = pd.read_msgpack(f'data/z6_ts_{split}_events.msgpack')
            df_discount = pd.read_msgpack(f'data/z6_ts_{split}_discount.msgpack')
        df = with_discount(df_events, df_discount)
        with TLOG('build index events'):
            events = IndexedEvents(df, key_column=self.key_column)
        return events

    def read_keys(self):
        if self.is_multikey:
            key_column = '_'.join(self.key_column)
        else:
            key_column = self.key_column
        filepath = 'data/z6_ts_{split}_{key_column}.npy'.format(
            split=self.split.name,
            key_column=key_column,
        )
        with TLOG('read keys'):
            keys = np.load(filepath)
        if self.is_multikey:
            keys = [tuple(x) for x in keys]
        else:
            keys = keys.tolist()
        return keys

    @profile
    def process(self):
        is_multikey = self.is_multikey
        events = self.read_events()
        keys = self.read_keys()
        extract_feature = self.extractor.extract
        result = []
        for key in tqdm.tqdm(keys, desc=self.name):
            sub_events = events.loc(key)
            idx_date = np.unique(sub_events['t'])
            for t in idx_date:
                row = extract_feature(events, key, t)
                if is_multikey:
                    result.append([*key, t, *row])
                else:
                    result.append([key, t, *row])
        df_result = pd.DataFrame.from_records(result, columns=self.columns)
        return df_result

    def save_result(self, df):
        LOG.info('result shape={}, from={}, to={}',
                 df.shape, format_date(df['date'].min()), format_date(df['date'].max()))
        filepath = 'data/z6_ts_{}_feature_{}.msgpack'.format(
            self.split.name, self.feature.name)
        with TLOG('save result'):
            df.to_msgpack(filepath)

    @staticmethod
    def main(feature, split):
        processor = FeatureProcessor(feature, split)
        df = processor.process()
        processor.save_result(df)


def date_of_days(days):
    return pd.Timestamp('2016-01-01') + pd.to_timedelta(days, unit='d')


def format_date(d):
    return date_of_days(d).date().isoformat()


def with_date_feature(df):
    t = date_of_days(df['date'])
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
    def merge_feature(cls, df, df_user, df_merchant, df_coupon, df_user_merchant, df_user_coupon):
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

        df = pd.merge(df, df_user_coupon, how='left',
                      left_on=['user_id', 'coupon_id', 'date'],
                      right_on=['user_id', 'coupon_id', 'date'])

        with_date_feature(df)
        df = df.reset_index(drop=True)
        return df

    @staticmethod
    def main(split):

        def read_dataframe(name):
            filepath = f'data/z6_ts_{split.name}_{name}.msgpack'
            return pd.read_msgpack(filepath)

        def save_dataframe(name, df):
            LOG.info(
                '{} shape={}, from={}, to={}',
                name, df.shape,
                df['date'].min().date().isoformat(),
                df['date'].max().date().isoformat()
            )
            filepath = f'data/z6_ts_{split.name}_merged_{name}.msgpack'
            return df.to_msgpack(filepath)

        with TLOG('read features'):
            df_user = read_dataframe('feature_user')
            df_merchant = read_dataframe('feature_merchant')
            df_coupon = read_dataframe('feature_coupon')
            df_user_merchant = read_dataframe('feature_user_merchant')
            df_user_coupon = read_dataframe('feature_user_coupon')
            df_discount = read_dataframe('discount')
            df_events = read_dataframe('events')
        feature_dfs = dict(
            df_user=df_user,
            df_merchant=df_merchant,
            df_coupon=df_coupon,
            df_user_merchant=df_user_merchant,
            df_user_coupon=df_user_coupon,
        )

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
        df = MergedFeature.merge_feature(
            with_discount(df_test, df_discount), **feature_dfs)
        save_dataframe('test', df)

        LOG.info('merge for train')
        df = MergedFeature.merge_feature(
            with_discount(df_train, df_discount), **feature_dfs)
        save_dataframe('train', df)


def get_split():
    split_name = get_config('SPLIT', 'validate')
    splits = [TestSplit, ValidateSplit]
    splits = {x.name: x for x in splits}
    return splits[split_name]


if __name__ == "__main__":
    split = get_split()
    LOG.info('split={}', split.name)
    O2OEvents.main(split=split)
    for feat in FEATURES:
        with TLOG(f'process {feat.name} feature'):
            FeatureProcessor.main(feat, split=split)
    with TLOG(f'merge features'):
        MergedFeature.main(split=split)
