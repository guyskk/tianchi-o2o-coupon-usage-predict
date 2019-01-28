import random
import itertools
import json
import datetime
import multiprocessing

import tqdm
import lightgbm as lgb
import pandas as pd
import numpy as np
from loguru import logger as LOG
from sklearn.metrics import roc_auc_score

from util import TLOG
from feature_extractor import get_split


SPLIT = get_split()
TEST_HAS_LABEL = SPLIT.test_has_label
LOG.info('TEST_HAS_LABEL={}', TEST_HAS_LABEL)
N_JOBS = max(multiprocessing.cpu_count() - 2, 1)


def pretty(data):
    return json.dumps(data, indent=4, ensure_ascii=False)


def split_feature_label(df):
    cols = list(df.columns)
    cols.remove('label')
    df_x = df[cols]
    df_y = df['label']
    return df_x, df_y


def format_date(df):
    df = df.copy()
    df['date'] = df.date.dt.strftime('%Y%m%d')
    return df


def get_now():
    return datetime.datetime.now().strftime('%Y%m%d-%H%M%S')


def o2o_auc_score(df):
    def coupon_auc(df):
        try:
            return roc_auc_score(df['label'], df['prob'])
        except ValueError:
            return None
    o2o_auc = df.groupby('coupon_id').apply(coupon_auc).mean()
    std_auc = roc_auc_score(df['label'], df['prob'])
    return o2o_auc, std_auc


def split_big_merchant(df, df_full):
    """
    将占总发券数80%的大商家取出来
    """
    mc = df_full['merchant_id'].value_counts()
    i = 5
    while True:
        p = mc.head(i).sum() / len(df_full)
        if p > 0.8:
            break
        i += 5
    min_coupons = mc.iloc[i]
    LOG.info('top {}/{} big merchants, at least {} coupons, percent {}'.format(i, len(mc), min_coupons, p))
    big_merchant_ids = mc.head(i).index.values
    big_mask = df_full['merchant_id'].isin(big_merchant_ids)
    df_big = df[big_mask]
    df_big_full = df_full[big_mask]
    df_small = df[~big_mask]
    df_small_full = df_full[~big_mask]
    ret = [df_big, df_big_full, df_small, df_small_full]
    LOG.info('big={}, big_full={}, small={}, small_full={}', *[x.shape for x in ret])
    return ret

GOOD_FEATURES = [
    "coupon:future.offline_receive_coupon.count",
        "coupon:history(offline_buy_with_coupon.count/offline_receive_coupon.count)",
        "coupon:history.offline_buy_with_coupon.distance.max",
        "coupon:history.offline_buy_with_coupon.distance.mean",
        "coupon:history.offline_buy_with_coupon.distance.min",
        "coupon:history.offline_receive_coupon.distance.mean",
        "coupon:history.offline_receive_coupon.distance.min",
        "coupon:history.offline_receive_coupon.unique_user.count",
        "coupon:longago.offline_buy_with_coupon.count",
        "coupon:longago.offline_buy_with_coupon.distance.max",
        "coupon:longago.offline_buy_with_coupon.unique_user.count",
        "coupon:longago.offline_receive_coupon.count",
        "coupon:longago.offline_receive_coupon.distance.max",
        "coupon:longago.offline_receive_coupon.distance.mean",
        "coupon:longago.offline_receive_coupon.unique_user.count",
        "coupon:recent.offline_buy_with_coupon.count",
        "coupon:recent.offline_buy_with_coupon.distance.mean",
        "coupon:recent.offline_buy_with_coupon.distance.min",
        "coupon:recent.offline_receive_coupon.count",
        "coupon:recent.offline_receive_coupon.distance.mean",
        "coupon:recent.offline_receive_coupon.distance.min",
        "coupon:recent.offline_receive_coupon.unique_user.count",
        "coupon:today.offline_receive_coupon.count",
        "date_dayofmonth",
        "date_dayofweek",
        "discount_jian",
        "discount_man",
        "discount_rate",
        "distance",
        "future.offline_receive_coupon.count",
        "is_xianshi",
        "merchant:future.offline_receive_coupon.unique_coupon.count",
        "merchant:future.offline_receive_coupon.unique_user.count",
        "merchant:history(offline_buy_with_coupon.count/offline_buy_with_coupon.unique_user.count)",
        "merchant:history(offline_buy_with_coupon.count/offline_receive_coupon.count)",
        "merchant:history.offline.unique_coupon.count",
        "merchant:history.offline.unique_hotcoupon.count",
        "merchant:history.offline.unique_user.count",
        "merchant:history.offline_buy_with_coupon.count",
        "merchant:history.offline_buy_with_coupon.discount_rate.min",
        "merchant:history.offline_buy_with_coupon.distance.max",
        "merchant:history.offline_buy_with_coupon.distance.mean",
        "merchant:history.offline_buy_with_coupon.distance.min",
        "merchant:history.offline_buy_with_coupon.timedelta.max",
        "merchant:history.offline_buy_with_coupon.timedelta.mean",
        "merchant:history.offline_buy_with_coupon.timedelta.min",
        "merchant:history.offline_buy_with_coupon.unique_hotuser.count",
        "merchant:history.offline_buy_with_coupon.unique_user.count",
        "merchant:history.offline_buy_without_coupon.count",
        "merchant:history.offline_buy_without_coupon.distance.max",
        "merchant:history.offline_buy_without_coupon.distance.mean",
        "merchant:history.offline_buy_without_coupon.distance.min",
        "merchant:history.offline_buy_without_coupon.unique_hotuser.count",
        "merchant:history.offline_buy_without_coupon.unique_user.count",
        "merchant:history.offline_receive_coupon.count",
        "merchant:history.offline_receive_coupon.discount_rate.max",
        "merchant:history.offline_receive_coupon.discount_rate.min",
        "merchant:longago(offline_buy_with_coupon.count/offline_buy_with_coupon.unique_user.count)",
        "merchant:longago(offline_buy_with_coupon.count/offline_receive_coupon.count)",
        "merchant:longago(offline_buy_with_coupon.unique_user.count/offline_receive_coupon.unique_user.count)",
        "merchant:longago.offline.unique_hotcoupon.count",
        "merchant:longago.offline.unique_user.count",
        "merchant:longago.offline_buy_with_coupon.count",
        "merchant:longago.offline_buy_with_coupon.discount_rate.max",
        "merchant:longago.offline_buy_with_coupon.discount_rate.min",
        "merchant:longago.offline_buy_with_coupon.distance.mean",
        "merchant:longago.offline_buy_with_coupon.distance.min",
        "merchant:longago.offline_buy_with_coupon.timedelta.max",
        "merchant:longago.offline_buy_with_coupon.timedelta.mean",
        "merchant:longago.offline_buy_with_coupon.timedelta.min",
        "merchant:longago.offline_buy_with_coupon.unique_hotuser.count",
        "merchant:longago.offline_buy_with_coupon.unique_user.count",
        "merchant:longago.offline_buy_without_coupon.count",
        "merchant:longago.offline_buy_without_coupon.distance.max",
        "merchant:longago.offline_buy_without_coupon.distance.mean",
        "merchant:longago.offline_buy_without_coupon.unique_hotuser.count",
        "merchant:longago.offline_buy_without_coupon.unique_user.count",
        "merchant:longago.offline_receive_coupon.count",
        "merchant:longago.offline_receive_coupon.discount_rate.mean",
        "merchant:longago.offline_receive_coupon.discount_rate.min",
        "merchant:longago.offline_receive_coupon.unique_user.count",
        "merchant:recent(offline_buy_with_coupon.count/offline_buy_with_coupon.unique_user.count)",
        "merchant:recent.offline.unique_coupon.count",
        "merchant:recent.offline.unique_hotcoupon.count",
        "merchant:recent.offline.unique_user.count",
        "merchant:recent.offline_buy_with_coupon.count",
        "merchant:recent.offline_buy_with_coupon.discount_rate.max",
        "merchant:recent.offline_buy_with_coupon.distance.max",
        "merchant:recent.offline_buy_with_coupon.distance.min",
        "merchant:recent.offline_buy_with_coupon.timedelta.max",
        "merchant:recent.offline_buy_with_coupon.timedelta.mean",
        "merchant:recent.offline_buy_with_coupon.timedelta.min",
        "merchant:recent.offline_buy_with_coupon.unique_user.count",
        "merchant:recent.offline_buy_without_coupon.count",
        "merchant:recent.offline_buy_without_coupon.distance.max",
        "merchant:recent.offline_buy_without_coupon.distance.min",
        "merchant:recent.offline_buy_without_coupon.unique_hotuser.count",
        "merchant:recent.offline_receive_coupon.discount_rate.max",
        "merchant:recent.offline_receive_coupon.discount_rate.min",
        "merchant:recent.offline_receive_coupon.unique_user.count",
        "merchant:today.offline_receive_coupon.count",
        "merchant:today.offline_receive_coupon.unique_coupon.count",
        "merchant:today.offline_receive_coupon.unique_user.count",
        "today.offline_receive_coupon.count",
        "user:future.offline_receive_coupon.count",
        "user:future.offline_receive_coupon.deltanow.min",
        "user:future.offline_receive_coupon.unique_coupon.count",
        "user:future.offline_receive_coupon.unique_merchant.count",
        "user:history(offline_buy_with_coupon.count/offline_buy_with_coupon.unique_merchant.count)",
        "user:history(offline_buy_with_coupon.count/offline_receive_coupon.count)",
        "user:history(offline_buy_with_coupon.man200.count/offline_buy_with_coupon.count)",
        "user:history(offline_buy_with_coupon.man200.count/offline_receive_coupon.man200.count)",
        "user:history(offline_buy_with_coupon.unique_coupon.count/offline.unique_coupon.count)",
        "user:history(offline_buy_with_coupon.unique_merchant.count/offline.unique_merchant.count)",
        "user:history(online_buy_with_coupon.count/online_receive_coupon.count)",
        "user:history(online_receive_coupon.count/online_click.count)",
        "user:history.offline.unique_coupon.count",
        "user:history.offline.unique_merchant.count",
        "user:history.offline_buy_with_coupon.count",
        "user:history.offline_buy_with_coupon.discount_rate.mean",
        "user:history.offline_buy_with_coupon.discount_rate.min",
        "user:history.offline_buy_with_coupon.distance.max",
        "user:history.offline_buy_with_coupon.distance.min",
        "user:history.offline_buy_with_coupon.man200.count",
        "user:history.offline_buy_with_coupon.timedelta.max",
        "user:history.offline_buy_with_coupon.timedelta.mean",
        "user:history.offline_buy_with_coupon.timedelta.min",
        "user:history.offline_buy_with_coupon.unique_coupon.count",
        "user:history.offline_buy_without_coupon.count",
        "user:history.offline_buy_without_coupon.distance.mean",
        "user:history.offline_receive_coupon.count",
        "user:history.offline_receive_coupon.discount_rate.mean",
        "user:history.offline_receive_coupon.discount_rate.min",
        "user:history.offline_receive_coupon.man200.count",
        "user:history.online_buy_with_coupon.count",
        "user:history.online_buy_without_coupon.count",
        "user:history.online_receive_coupon.count",
        "user:longago(offline_buy_with_coupon.count/offline_buy_with_coupon.unique_merchant.count)",
        "user:longago(offline_buy_with_coupon.count/offline_receive_coupon.count)",
        "user:longago(offline_buy_with_coupon.man200.count/offline_buy_with_coupon.count)",
        "user:longago(offline_buy_with_coupon.unique_coupon.count/offline.unique_coupon.count)",
        "user:longago(offline_buy_with_coupon.unique_merchant.count/offline.unique_merchant.count)",
        "user:longago.offline.unique_coupon.count",
        "user:longago.offline_buy_with_coupon.count",
        "user:longago.offline_buy_with_coupon.discount_rate.max",
        "user:longago.offline_buy_with_coupon.distance.max",
        "user:longago.offline_buy_with_coupon.distance.mean",
        "user:longago.offline_buy_with_coupon.distance.min",
        "user:longago.offline_buy_with_coupon.man200.count",
        "user:longago.offline_buy_with_coupon.timedelta.min",
        "user:longago.offline_buy_with_coupon.unique_coupon.count",
        "user:longago.offline_buy_with_coupon.unique_merchant.count",
        "user:longago.offline_buy_without_coupon.count",
        "user:longago.offline_buy_without_coupon.distance.max",
        "user:longago.offline_buy_without_coupon.distance.mean",
        "user:longago.offline_receive_coupon.count",
        "user:longago.offline_receive_coupon.discount_rate.max",
        "user:longago.offline_receive_coupon.discount_rate.mean",
        "user:longago.offline_receive_coupon.discount_rate.min",
        "user:longago.offline_receive_coupon.man200.count",
        "user:recent(offline_buy_with_coupon.man200.count/offline_receive_coupon.man200.count)",
        "user:recent(offline_buy_with_coupon.unique_coupon.count/offline.unique_coupon.count)",
        "user:recent(online_buy_with_coupon.count/online_receive_coupon.count)",
        "user:recent.offline.unique_coupon.count",
        "user:recent.offline.unique_merchant.count",
        "user:recent.offline_buy_with_coupon.discount_rate.max",
        "user:recent.offline_buy_with_coupon.discount_rate.mean",
        "user:recent.offline_buy_with_coupon.discount_rate.min",
        "user:recent.offline_buy_with_coupon.distance.max",
        "user:recent.offline_buy_with_coupon.distance.mean",
        "user:recent.offline_buy_with_coupon.distance.min",
        "user:recent.offline_buy_with_coupon.man200.count",
        "user:recent.offline_buy_with_coupon.unique_coupon.count",
        "user:recent.offline_buy_with_coupon.unique_merchant.count",
        "user:recent.offline_buy_without_coupon.count",
        "user:recent.offline_buy_without_coupon.distance.max",
        "user:recent.offline_buy_without_coupon.distance.mean",
        "user:recent.offline_buy_without_coupon.distance.min",
        "user:recent.offline_receive_coupon.count",
        "user:recent.offline_receive_coupon.discount_rate.max",
        "user:recent.offline_receive_coupon.discount_rate.mean",
        "user:recent.offline_receive_coupon.discount_rate.min",
        "user:recent.offline_receive_coupon.man200.count",
        "user:recent.online_buy_without_coupon.count",
        "user:today.offline_receive_coupon.count",
        "user_merchant:future.offline_receive_coupon.deltanow.min",
        "user_merchant:history(offline_buy_with_coupon.count/offline_receive_coupon.count)",
        "user_merchant:history(offline_receive_coupon.count-offline_buy_with_coupon.count)",
        "user_merchant:longago(offline_buy_with_coupon.count/offline_receive_coupon.count)",
        "user_merchant:longago(offline_receive_coupon.count-offline_buy_with_coupon.count)",
        "user_merchant:longago.offline_buy_without_coupon.count",
        "user_merchant:longago.offline_receive_coupon.count",
        "user_merchant:recent(offline_receive_coupon.count-offline_buy_with_coupon.count)",
        "user_merchant:recent.offline_buy_without_coupon.count",
        "user_merchant:today.offline_receive_coupon.count",
        "coupon:history.offline_buy_with_coupon.distance.mean",
        "coupon:history.offline_receive_coupon.distance.min",
        "coupon:longago.offline_buy_with_coupon.timedelta.mean",
        "coupon:longago.offline_buy_with_coupon.unique_user.count",
        "coupon:recent.offline_receive_coupon.count",
        "coupon:recent.offline_receive_coupon.distance.max",
        "date_dayofmonth",
        "date_dayofweek",
        "discount_jian",
        "discount_man",
        "discount_rate",
        "future.offline_receive_coupon.count",
        "future.offline_receive_coupon.deltanow.min",
        "is_manjian",
        "merchant:future.offline_receive_coupon.unique_coupon.count",
        "merchant:future.offline_receive_coupon.unique_user.count",
        "merchant:history(offline_buy_with_coupon.count/offline_buy_with_coupon.unique_user.count)",
        "merchant:history.offline.unique_user.count",
        "merchant:history.offline_buy_with_coupon.count",
        "merchant:history.offline_buy_with_coupon.timedelta.min",
        "merchant:history.offline_buy_without_coupon.count",
        "merchant:history.offline_buy_without_coupon.distance.mean",
        "merchant:history.offline_buy_without_coupon.unique_hotuser.count",
        "merchant:history.offline_receive_coupon.count",
        "merchant:history.offline_receive_coupon.unique_user.count",
        "merchant:longago.offline.unique_hotcoupon.count",
        "merchant:longago.offline.unique_user.count",
        "merchant:longago.offline_buy_with_coupon.count",
        "merchant:longago.offline_buy_with_coupon.discount_rate.max",
        "merchant:longago.offline_buy_with_coupon.distance.mean",
        "merchant:longago.offline_buy_with_coupon.distance.min",
        "merchant:longago.offline_buy_with_coupon.timedelta.max",
        "merchant:longago.offline_buy_without_coupon.count",
        "merchant:longago.offline_buy_without_coupon.unique_hotuser.count",
        "merchant:longago.offline_buy_without_coupon.unique_user.count",
        "merchant:longago.offline_receive_coupon.count",
        "merchant:longago.offline_receive_coupon.discount_rate.min",
        "merchant:recent(offline_buy_with_coupon.count/offline_receive_coupon.count)",
        "merchant:recent(offline_buy_with_coupon.unique_user.count/offline_receive_coupon.unique_user.count)",
        "merchant:recent.offline.unique_user.count",
        "merchant:recent.offline_buy_with_coupon.distance.min",
        "merchant:recent.offline_buy_with_coupon.timedelta.max",
        "merchant:recent.offline_buy_with_coupon.unique_user.count",
        "merchant:recent.offline_buy_without_coupon.count",
        "merchant:recent.offline_buy_without_coupon.distance.mean",
        "merchant:recent.offline_buy_without_coupon.unique_hotuser.count",
        "merchant:recent.offline_buy_without_coupon.unique_user.count",
        "merchant:recent.offline_receive_coupon.count",
        "merchant:recent.offline_receive_coupon.discount_rate.mean",
        "merchant:recent.offline_receive_coupon.unique_user.count",
        "merchant:today.offline_receive_coupon.count",
        "merchant:today.offline_receive_coupon.unique_user.count",
        "today.offline_receive_coupon.count",
        "user:future.offline_receive_coupon.count",
        "user:future.offline_receive_coupon.deltanow.min",
        "user:future.offline_receive_coupon.unique_coupon.count",
        "user:history(offline_buy_with_coupon.count/offline_buy_with_coupon.unique_merchant.count)",
        "user:history(offline_buy_with_coupon.count/offline_receive_coupon.count)",
        "user:history(offline_buy_with_coupon.man200.count/offline_buy_with_coupon.count)",
        "user:history.offline_buy_with_coupon.count",
        "user:history.offline_buy_with_coupon.discount_rate.max",
        "user:history.offline_buy_without_coupon.count",
        "user:history.offline_receive_coupon.discount_rate.mean",
        "user:history.offline_receive_coupon.man200.count",
        "user:history.online_buy_without_coupon.count",
        "user:longago(offline_buy_with_coupon.man200.count/offline_receive_coupon.man200.count)",
        "user:longago.offline_buy_with_coupon.distance.min",
        "user:longago.offline_buy_with_coupon.man200.count",
        "user:longago.offline_buy_with_coupon.timedelta.mean",
        "user:longago.offline_receive_coupon.discount_rate.max",
        "user:longago.offline_receive_coupon.discount_rate.mean",
        "user:longago.offline_receive_coupon.man200.count",
        "user:recent(offline_buy_with_coupon.man200.count/offline_buy_with_coupon.count)",
        "user:recent(offline_buy_with_coupon.man200.count/offline_receive_coupon.man200.count)",
        "user:recent.offline.unique_coupon.count",
        "user:recent.offline.unique_merchant.count",
        "user:recent.offline_buy_with_coupon.count",
        "user:recent.offline_buy_with_coupon.discount_rate.max",
        "user:recent.offline_buy_with_coupon.timedelta.max",
        "user:recent.offline_buy_without_coupon.count",
        "user:recent.offline_receive_coupon.discount_rate.mean",
        "user:recent.offline_receive_coupon.discount_rate.min",
        "user:recent.online_buy_without_coupon.count",
        "user:recent.online_receive_coupon.count",
        "user:today.offline_receive_coupon.count",
        "user_merchant:future.offline_receive_coupon.count",
        "user_merchant:future.offline_receive_coupon.deltanow.min",
        "user_merchant:history.offline_buy_with_coupon.count",
        "user_merchant:longago.offline_buy_without_coupon.count",
        "user_merchant:longago.offline_receive_coupon.count",
        "user_merchant:recent(offline_receive_coupon.count-offline_buy_with_coupon.count)",
        "user_merchant:recent.offline_buy_with_coupon.count",
        "user_merchant:recent.offline_buy_without_coupon.count",
]

GOOD_FEATURES = list(sorted(set(GOOD_FEATURES)))


def read_dataset():
    with TLOG('read dataframes'):
        df_test_full = pd.read_msgpack(f'data/z6_ts_{SPLIT.name}_merged_test.msgpack')
        df_train_full = pd.read_msgpack(f'data/z6_ts_{SPLIT.name}_merged_train.msgpack')
    # columns
    features = list(df_test_full.columns.difference([
        'user_id', 'merchant_id', 'coupon_id',
        'discount_name', 'date', 'label',
    ]))
    features = GOOD_FEATURES
    # features = list(
    #     set(itertools.chain(*FEATULE_LEVELS.values()))
    #     - set(BAD_FEATURES)
    # )
    print(pretty(features))
    test_submit_cols = ['user_id', 'coupon_id', 'date']
    if TEST_HAS_LABEL:
        test_submit_cols += ['label']
    train_submit_cols = ['user_id', 'coupon_id', 'date', 'label']
    # test submit
    df_test = df_test_full[features]
    df_submit = format_date(df_test_full.loc[df_test.index, test_submit_cols])
    LOG.info('df_test {}', df_test.shape)
    LOG.info('df_submit {}', df_submit.shape)
    # split train validate
    mask = np.random.rand(len(df_train_full)) < 0.05
    df_validate = df_train_full.loc[mask, features + ['label']]
    df_validate_submit = format_date(df_train_full.loc[mask, train_submit_cols])
    df_train = df_train_full.loc[~mask, features + ['label']]
    df_train_submit = format_date(df_train_full.loc[~mask, train_submit_cols])
    LOG.info('df_train {}', df_train.shape)
    LOG.info('df_train_submit {}', df_train_submit.shape)
    LOG.info('df_validate {}', df_validate.shape)
    LOG.info('df_validate_submit {}', df_validate_submit.shape)

    df_train_x, df_train_y = split_feature_label(df_train)
    df_validate_x, df_validate_y = split_feature_label(df_validate)

    ret = df_train_x, df_train_y, df_validate_x, df_validate_y, df_test, df_submit, df_validate_submit, df_train_submit
    return [x.copy() for x in ret]


(
    df_train_x,
    df_train_y,
    df_validate_x,
    df_validate_y,
    df_test,
    df_submit,
    df_validate_submit,
    df_train_submit,
) = read_dataset()
SUBMIT_NAME = get_now()


def build_params():
    params_combines = [
        ('max_depth', [2, 4, 8, 16]),
        ('max_leaves', [4, 8, 16, 32, 64]),
        ('feature_fraction', [0.2, 0.5, 1.0]),
        ('scale_pos_weight', [2, 4, 8]),
        ('lambda_l1', [0.1, 1, 10]),
        ('lambda_l2', [0.1, 1, 10]),
        ('eta', [0.1, 0.01, 0.003]),
    ]
    params_overides = []
    names = [x[0] for x in params_combines]
    for ps in itertools.product(*[x[1] for x in params_combines]):
        params_overide = list(zip(names, ps))
        params_overides.append(params_overide)
    return params_overides


def predict(model, df, df_submit):
    df_submit = df_submit.copy()
    df_submit['prob'] = model.predict(df)
    return df_submit


def predict_score(model, df, df_submit, return_submit=False):
    df_submit = predict(model, df, df_submit)
    o2o_score, std_score = o2o_auc_score(df_submit)
    if return_submit:
        return df_submit, o2o_score, std_score
    else:
        return o2o_score, std_score


def build_feature_groups(features):
    basic_features = []
    user_features = []
    merchant_features = []
    coupon_features = []
    user_merchant_features = []
    longago_features = []
    history_features = []
    recent_features = []
    today_features = []
    future_features = []
    for feat in features:
        if ':' not in feat:
            basic_features.append(feat)
        if 'user:' in feat:
            user_features.append(feat)
        if 'merchant:' in feat:
            merchant_features.append(feat)
        if 'coupon:' in feat:
            coupon_features.append(feat)
        if 'user_merchant:' in feat:
            user_merchant_features.append(feat)
        if 'longago' in feat:
            longago_features.append(feat)
        if 'history' in feat:
            history_features.append(feat)
        if 'recent' in feat:
            recent_features.append(feat)
        if 'today' in feat:
            today_features.append(feat)
        if 'future' in feat:
            future_features.append(feat)
    return [
        ('basic', basic_features),
        ('user', user_features),
        ('merchant', merchant_features),
        ('coupon', coupon_features),
        ('user_merchant', user_merchant_features),
        ('longago', longago_features),
        ('history', history_features),
        ('recent', recent_features),
        ('today', today_features),
        ('future', future_features),
    ]


def build_feature_cluster(features, n=6):
    features = list(features)
    random.shuffle(features)
    clusters = [(str(i), []) for i in range(1, n + 1)]
    for i, x in enumerate(features):
        clusters[i % n][1].append(x)
    assert sum([len(x[1]) for x in clusters]) == len(features), clusters
    return clusters


class FeatureSelector:
    def __init__(self, features):
        self.features = list(features)
        self.columns = [
            'cluster', 'feature', 'test_o2o_score', 'test_std_score',
            'validate_o2o_score', 'validate_std_score',
            'train_o2o_score', 'train_std_score',
        ]
        self.params = {
            'objective': 'binary',
            'metric': 'auc',
            'learning_rate': 0.03,
            'scale_pos_weight': 4,
            'max_depth': 16,
            'max_leaves': 32,
            'num_threads': 4,
            'num_trees': 5000,
            'early_stopping_rounds': 100,
        }

    def compute_cluster(self):
        feature_cluster = build_feature_cluster(self.features)
        for cluster, cluster_features in feature_cluster:
            for feature in tqdm.tqdm(cluster_features, desc=cluster):
                features = list(cluster_features)
                features.remove(feature)
                scores = self.score_features(f'{cluster},{feature}', features)
                scores = ','.join(['{:.6f}'.format(x) for x in scores])
                line = f'{cluster},{feature},{scores}\n'
                print(line)
                with open('feature_scores_by_cluster.csv', 'a') as f:
                    f.write(line)

    def read_feature_scores(self, filepath):
        df_scores = pd.read_csv(filepath, header=None, names=self.columns)
        return df_scores

    def score_features(self, name, features):
        data_train = lgb.Dataset(df_train_x[features], label=df_train_y)
        data_validate = lgb.Dataset(df_validate_x[features], label=df_validate_y)
        eval_params = dict(
            valid_sets=[data_train, data_validate],
            valid_names=['train', 'validate'],
            verbose_eval=50,
        )
        with TLOG(f'{name} train'):
            model = lgb.train(self.params, data_train, **eval_params)
        with TLOG(f'{name} predict'):
            scores = [
                *predict_score(model, df_test[features].head(10000), df_submit.head(10000)),
                *predict_score(model, df_validate_x[features], df_validate_submit),
                *predict_score(model, df_train_x[features], df_train_submit),
            ]
        return scores

    @staticmethod
    def select_low_features(df, col, sigma=1):
        low = df[col].mean() - sigma * df[col].std()
        feats = df[df[col] < low]['feature']
        return set(feats)

    @staticmethod
    def select_high_features(df, col, sigma=1):
        high = df[col].mean() + sigma * df[col].std()
        feats = df[df[col] > high]['feature']
        return set(feats)

    def get_cluster_features(self):
        df_scores = self.read_feature_scores('feature_scores_by_cluster.csv')
        groups = list(df_scores.groupby('cluster'))
        features = set()
        for i, (cluster, df) in enumerate(groups):
            for col in self.columns[2:]:
                features.update(self.select_low_features(df, col))
        LOG.info('cluster features:\n{}', pretty(list(sorted(features))))
        return features

    def compute_base(self):
        cluster_features = self.get_cluster_features()
        cluster = 'base'
        for feature in tqdm.tqdm(cluster_features, desc=cluster):
            features = list(cluster_features)
            features.remove(feature)
            scores = self.score_features(f'{cluster},{feature}', features)
            scores = ','.join(['{:.6f}'.format(x) for x in scores])
            line = f'{cluster},{feature},{scores}\n'
            print(line)
            with open('feature_scores_by_base.csv', 'a') as f:
                f.write(line)

    def get_base_features(self):
        df_scores = self.read_feature_scores('feature_scores_by_base.csv')
        good_features = self.select_low_features(df_scores, 'test_o2o_score', sigma=0)
        bad_features = set()
        for col in [
            'test_std_score', 'validate_o2o_score', 'validate_std_score',
            'train_o2o_score', 'train_std_score',
        ]:
            feats = self.select_low_features(df_scores, col)
            bad_features.update(feats - good_features)
        cluster_features = self.get_cluster_features()
        other_features = cluster_features - good_features - bad_features
        LOG.info('good features:\n{}', pretty(list(sorted(good_features))))
        LOG.info('bad features:\n{}', pretty(list(sorted(bad_features))))
        LOG.info('other features:\n{}', pretty(list(sorted(other_features))))
        return good_features, bad_features, other_features

    def compute_other(self):
        good_features, bad_features, other_features = self.get_base_features()
        fixed_features = good_features | bad_features | other_features
        candidate_features = list(set(self.features) - fixed_features)
        cluster = 'other'
        for feature in tqdm.tqdm(candidate_features, desc=cluster):
            features = list(fixed_features) + [feature]
            scores = self.score_features(f'{cluster},{feature}', features)
            scores = ','.join(['{:.6f}'.format(x) for x in scores])
            line = f'{cluster},{feature},{scores}\n'
            print(line)
            with open('feature_scores_by_other.csv', 'a') as f:
                f.write(line)

    def get_other_features(self):
        df_scores = self.read_feature_scores('feature_scores_by_other.csv')
        good_features = self.select_high_features(df_scores, 'test_o2o_score', sigma=0)
        bad_features = set()
        for col in [
            'test_std_score', 'validate_o2o_score', 'validate_std_score',
            'train_o2o_score', 'train_std_score',
        ]:
            feats = self.select_high_features(df_scores, col)
            bad_features.update(feats - good_features)
        other_features = set(self.features) - good_features - bad_features
        LOG.info('good features:\n{}', pretty(list(sorted(good_features))))
        LOG.info('bad features:\n{}', pretty(list(sorted(bad_features))))
        LOG.info('other features:\n{}', pretty(list(sorted(other_features))))
        return good_features, bad_features, other_features

    def save_feature_levels(self):
        good_features, bad_features, other_features = self.get_base_features()
        good_features2, bad_features2, other_features2 = self.get_other_features()
        result = {
            'good': list(sorted(good_features | good_features2)),
            'bad': list(sorted(bad_features | bad_features2)),
            'other': list(sorted(other_features | other_features2)),
        }
        content = pretty(result)
        print(content)
        with open('feature_levels.json', 'w') as f:
            f.write(content)

    @staticmethod
    def main():
        selector = FeatureSelector(df_test.columns)
        # selector.compute_cluster()
        # selector.compute_base()
        # selector.compute_other()
        selector.save_feature_levels()


def main():
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': 0.01,
        'feature_fraction': 0.2,
        'scale_pos_weight': 4,
        'max_depth': 8,
        'max_leaves': 16,
        'num_threads': 4,
        'num_trees': 5000,
        'early_stopping_rounds': 100,
    }
    data_train = lgb.Dataset(df_train_x, label=df_train_y)
    data_validate = lgb.Dataset(df_validate_x, label=df_validate_y)
    eval_params = dict(
        valid_sets=[data_train, data_validate],
        valid_names=['train', 'validate'],
        verbose_eval=50,
    )
    with TLOG('train'):
        model = lgb.train(params, data_train, **eval_params)
    filepath = f'data/model-lgb-{SUBMIT_NAME}.dat'
    LOG.info('save model {}', filepath)
    model.save_model(filepath)

    def save_and_score_submit(name, df, df_submit):
        submit_path = f'data/submit-lgb-{name}-{SUBMIT_NAME}.csv'
        LOG.info('save {} result {}', name, submit_path)
        if name == 'test' and not TEST_HAS_LABEL:
            df_sub = predict(model, df, df_submit)
        else:
            df_sub, o2o_score, std_score = predict_score(model, df, df_submit, return_submit=True)
            print('{} o2o auc: {:.3f}'.format(name, o2o_score))
        df_sub.to_csv(submit_path, index=False, header=False)

    save_and_score_submit('test', df_test, df_submit)
    save_and_score_submit('validate', df_validate_x, df_validate_submit)
    save_and_score_submit('train', df_train_x, df_train_submit)


if __name__ == "__main__":
    # FeatureSelector.main()
    main()
