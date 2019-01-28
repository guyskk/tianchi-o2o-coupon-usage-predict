import math

import numpy as np

from util import profile
from feature_definition import DATE_RANGES


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
