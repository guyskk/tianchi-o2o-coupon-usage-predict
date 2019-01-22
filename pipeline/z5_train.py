import sys
import json
import datetime
import lightgbm as lgb
import xgboost as xgb
import pandas as pd
import numpy as np
from loguru import logger as LOG
from sklearn.metrics import roc_auc_score

TEST_HAS_LABEL = False


def pretty(data):
    return json.dumps(data, indent=4, ensure_ascii=False)


def split_feature_label(df):
    df_x = df[df.columns.difference(['label'])]
    df_y = df['label']
    return df_x, df_y


def o2o_auc_score(df):
    def coupon_auc(df):
        try:
            return roc_auc_score(df['label'], df['prob'])
        except ValueError:
            return None
    return df.groupby('coupon_id').apply(coupon_auc).mean()


FEATURE_TOP = [
    'user_merchant:future.offline_receive_coupon.deltanow.min',
    'merchant:future.offline_receive_coupon.count',
    'merchant:recent.offline_buy_without_coupon.unique_hotuser.count',
    'coupon:future.offline_receive_coupon.count',
    'merchant:future.offline_receive_coupon.unique_user.count',
    'user:future.offline_receive_coupon.deltanow.min',
    'merchant:recent.offline_buy_without_coupon.distance.mean',
    'coupon:today.offline_receive_coupon.count',
    'discount_rate',
    'merchant:history.offline_buy_without_coupon.unique_hotuser.count',
    'merchant:recent.offline_buy_without_coupon.count',
    'merchant:recent.offline.unique_user.count',
    'merchant:longago.offline_buy_without_coupon.unique_hotuser.count',
    'merchant:today.offline_receive_coupon.count',
    'merchant:recent.offline_buy_without_coupon.unique_user.count',
    'merchant:history.offline_buy_without_coupon.distance.mean',
    'merchant:longago.offline_buy_without_coupon.count',
    'merchant:history.offline.unique_user.count',
    'date_dayofmonth',
    'merchant:longago.offline_buy_without_coupon.distance.mean',
    'user:future.offline_receive_coupon.count',
    'merchant:history.offline_buy_without_coupon.count',
    'user_merchant:future.offline_receive_coupon.count',
    'merchant:today.offline_receive_coupon.unique_user.count',
    'merchant:history.offline_buy_without_coupon.unique_user.count',
    'discount_man',
    'merchant:longago.offline.unique_user.count',
    'merchant:longago.offline_buy_without_coupon.unique_user.count',
    'merchant:recent(offline_buy_with_coupon.count/offline_buy_with_coupon.unique_user.count)',
    'merchant:recent.offline_receive_coupon.count',
    'user_merchant:recent.offline_buy_without_coupon.count',
    'merchant:recent.offline_receive_coupon.unique_user.count',
    'merchant:recent.offline_receive_coupon.discount_rate.mean',
    'merchant:history(offline_buy_with_coupon.unique_user.count/offline_receive_coupon.unique_user.count)',
    'merchant:recent(offline_buy_with_coupon.unique_user.count/offline_receive_coupon.unique_user.count)',
    'coupon:recent.offline_receive_coupon.unique_user.count',
    'merchant:recent.offline_buy_with_coupon.timedelta.mean',
    'merchant:recent(offline_buy_with_coupon.count/offline_receive_coupon.count)',
    'user_merchant:today.offline_receive_coupon.count',
    'coupon:recent.offline_receive_coupon.count',
    'distance',
    'merchant:history.offline_receive_coupon.unique_user.count',
    'discount_jian',
    'user:recent.offline_buy_without_coupon.count',
    'merchant:history(offline_buy_with_coupon.count/offline_receive_coupon.count)',
    'merchant:history.offline_receive_coupon.count',
    'merchant:history(offline_buy_with_coupon.count/offline_buy_with_coupon.unique_user.count)',
    'merchant:recent.offline_buy_with_coupon.count',
    'merchant:history.offline_buy_with_coupon.timedelta.mean',
    'user:history.offline_buy_without_coupon.count',
    'merchant:recent.offline_buy_with_coupon.unique_user.count',
    'user:today.offline_receive_coupon.count',
    'user:longago.offline_buy_without_coupon.count',
    'coupon:recent(offline_buy_with_coupon.count/offline_receive_coupon.count)',
    'merchant:history.offline_buy_with_coupon.unique_user.count',
    'user_merchant:longago.offline_buy_without_coupon.count',
    'merchant:longago(offline_buy_with_coupon.count/offline_buy_with_coupon.unique_user.count)',
    'user:recent.offline_receive_coupon.discount_rate.mean',
    'user_merchant:history.offline_buy_without_coupon.count',
    'merchant:longago(offline_buy_with_coupon.count/offline_receive_coupon.count)',
    'merchant:history.offline_buy_with_coupon.count',
    'date_dayofweek',
    'coupon:recent.offline_buy_with_coupon.timedelta.mean',
    'merchant:longago.offline_receive_coupon.count',
    'merchant:recent.offline_buy_with_coupon.timedelta.max',
    'merchant:longago.offline_receive_coupon.unique_user.count',
    'merchant:recent.offline_buy_with_coupon.unique_hotuser.count',
    'merchant:recent.offline_buy_with_coupon.discount_rate.mean',
    'merchant:history.offline_receive_coupon.discount_rate.mean',
    'merchant:longago(offline_buy_with_coupon.unique_user.count/offline_receive_coupon.unique_user.count)',
    'merchant:longago.offline_buy_with_coupon.timedelta.mean',
    'user:history.online_click.count',
    'user:future.offline_receive_coupon.unique_merchant.count',
    'merchant:history.offline_buy_with_coupon.discount_rate.mean',
    'merchant:history.offline_buy_without_coupon.distance.max',
    'user:longago.offline_receive_coupon.discount_rate.mean',
    'merchant:longago.offline_buy_with_coupon.unique_user.count',
    'merchant:recent.offline_buy_with_coupon.distance.mean',
    'user:recent.online_click.count',
    'user:history.offline_buy_with_coupon.timedelta.mean',
    'user:longago.offline.unique_merchant.count',
    'coupon:history.offline_receive_coupon.unique_user.count',
    'merchant:history.offline_buy_with_coupon.timedelta.max',
    'merchant:longago.offline_buy_without_coupon.distance.max',
    'merchant:history.offline_buy_with_coupon.distance.mean',
    'merchant:recent.offline_receive_coupon.discount_rate.min',
    'merchant:history.offline_buy_with_coupon.timedelta.min',
    'merchant:history.offline_buy_with_coupon.unique_hotuser.count',
    'coupon:recent.offline_receive_coupon.distance.mean',
    'merchant:longago.offline_buy_with_coupon.count',
    'user:history.online_buy_without_coupon.count',
    'user:history.offline_receive_coupon.discount_rate.mean',
    'coupon:recent.offline_buy_with_coupon.unique_user.count',
    'user:recent.online_receive_coupon.count',
    'user:history.offline.unique_merchant.count',
    'merchant:recent.offline_receive_coupon.discount_rate.max',
    'merchant:longago.offline_buy_with_coupon.distance.mean',
    'merchant:history.offline_buy_with_coupon.discount_rate.min',
    'merchant:recent.offline_buy_without_coupon.distance.max',
    'user_merchant:recent(offline_receive_coupon.count-offline_buy_with_coupon.count)',
    'coupon:recent.offline_buy_with_coupon.count',
    'user:recent.offline_buy_without_coupon.distance.mean',
    'user:history.online_receive_coupon.count',
    'user:recent.online_buy_without_coupon.count',
    'user:history.offline_buy_without_coupon.distance.mean',
    'user:recent.offline.unique_merchant.count',
    'coupon:history.offline_receive_coupon.count',
    'merchant:recent.offline_buy_with_coupon.discount_rate.max',
    'merchant:recent.offline_buy_with_coupon.discount_rate.min',
    'merchant:recent.offline_buy_with_coupon.timedelta.min',
    'coupon:longago.offline_receive_coupon.count',
    'user:recent.offline_receive_coupon.discount_rate.min',
    'user:recent.offline_receive_coupon.discount_rate.max',
    'user:longago.offline_buy_without_coupon.distance.mean',
    'user:recent.offline_receive_coupon.count',
    'user:history.offline_receive_coupon.discount_rate.min',
    'merchant:history.offline_receive_coupon.discount_rate.min',
    'user:history(online_receive_coupon.count/online_click.count)',
    'user:future.offline_receive_coupon.unique_coupon.count',
    'coupon:history.offline_receive_coupon.distance.mean',
    'user:recent.offline_buy_with_coupon.timedelta.mean',
    'user:longago.offline_buy_without_coupon.distance.max',
    'merchant:longago.offline_receive_coupon.discount_rate.mean',
    'user:history.offline_receive_coupon.discount_rate.max',
    'user_merchant:history(offline_receive_coupon.count-offline_buy_with_coupon.count)',
    'user:recent.offline_buy_without_coupon.distance.min',
    'merchant:longago.offline_buy_with_coupon.discount_rate.mean',
    'user:history.offline_receive_coupon.count',
    'merchant:history.offline_receive_coupon.discount_rate.max',
    'user:longago.offline_receive_coupon.discount_rate.min',
    'user:longago.offline_receive_coupon.count',
    'merchant:longago.offline_buy_with_coupon.timedelta.max',
    'user:longago.offline_buy_without_coupon.distance.min',
    'user:recent.offline_buy_with_coupon.timedelta.max',
    'user:recent.offline_buy_without_coupon.distance.max',
    'coupon:recent.offline_buy_with_coupon.distance.mean',
    'user:recent(offline_buy_with_coupon.count/offline_receive_coupon.count)',
    'merchant:history.offline_buy_with_coupon.discount_rate.max',
    'merchant:longago.offline_buy_with_coupon.timedelta.min',
    'user:recent.offline_buy_with_coupon.timedelta.min',
    'coupon:history(offline_buy_with_coupon.count/offline_receive_coupon.count)',
    'merchant:recent.offline_buy_with_coupon.distance.max',
    'coupon:recent.offline_receive_coupon.distance.max',
    'user_merchant:recent.offline_receive_coupon.count',
    'coupon:longago.offline_receive_coupon.distance.mean',
    'merchant:longago.offline_buy_with_coupon.unique_hotuser.count',
    'user:history(offline_buy_with_coupon.count/offline_receive_coupon.count)',
    'user:history.offline_buy_without_coupon.distance.max',
    'user:history.offline_buy_without_coupon.distance.min',
    'user:longago.offline_receive_coupon.discount_rate.max',
    'user:recent.offline_receive_coupon.man200.count',
    'user:longago.offline_buy_with_coupon.timedelta.min',
    'user_merchant:recent(offline_buy_with_coupon.count/offline_receive_coupon.count)',
    'merchant:longago.offline_receive_coupon.discount_rate.min',
    'user:history.offline_buy_with_coupon.timedelta.max',
    'merchant:longago.offline_receive_coupon.discount_rate.max',
    'merchant:recent.offline_buy_without_coupon.distance.min',
    'user:recent(online_receive_coupon.count/online_click.count)',
    'merchant:recent.offline_buy_with_coupon.distance.min',
    'user:recent.offline_buy_with_coupon.discount_rate.max',
    'user:history.offline_buy_with_coupon.timedelta.min',
    'user_merchant:history.offline_buy_with_coupon.count',
    'user:longago.offline_buy_with_coupon.timedelta.mean',
    'coupon:longago.offline_receive_coupon.unique_user.count',
    'merchant:history.offline_buy_with_coupon.distance.max',
    'user:recent.online_buy_with_coupon.count',
    'merchant:history.offline_buy_without_coupon.distance.min',
    'user:longago.offline_receive_coupon.man200.count',
    'user:longago(offline_buy_with_coupon.count/offline_receive_coupon.count)',
    'merchant:longago.offline_buy_with_coupon.discount_rate.min',
    'merchant:longago.offline_buy_without_coupon.distance.min',
    'user:recent.offline_buy_with_coupon.discount_rate.mean',
    'user_merchant:longago(offline_receive_coupon.count-offline_buy_with_coupon.count)',
    'merchant:longago.offline_buy_with_coupon.discount_rate.max',
    'coupon:history.offline_buy_with_coupon.count',
    'user_merchant:history.offline_receive_coupon.count',
    'user:recent.offline_buy_with_coupon.count',
    'is_manjian',
    'user:recent.offline.unique_coupon.count',
    'user:history.offline_receive_coupon.man200.count',
    'user:recent(offline_buy_with_coupon.count/offline_buy_with_coupon.unique_merchant.count)',
    'merchant:longago.offline_buy_with_coupon.distance.max',
    'user:recent(online_buy_with_coupon.count/online_receive_coupon.count)',
    'is_dazhe',
    'user:history.offline.unique_coupon.count',
    'user_merchant:recent.offline_buy_with_coupon.count',
    'user:longago.offline_buy_with_coupon.timedelta.max',
    'user:recent.offline_buy_with_coupon.discount_rate.min',
    'user:longago.offline_buy_with_coupon.discount_rate.min',
    'user:history.offline_buy_with_coupon.discount_rate.mean',
    'user:longago.offline.unique_coupon.count',
    'user:longago.offline_buy_with_coupon.discount_rate.mean',
    'coupon:longago(offline_buy_with_coupon.count/offline_receive_coupon.count)',
    'coupon:history.offline_buy_with_coupon.timedelta.mean',
    'user:history(online_buy_with_coupon.count/online_receive_coupon.count)',
    'user:longago(offline_buy_with_coupon.count/offline_buy_with_coupon.unique_merchant.count)',
    'merchant:future.offline_receive_coupon.unique_coupon.count',
    'user:history.offline_buy_with_coupon.count',
    'user:history(offline_buy_with_coupon.count/offline_buy_with_coupon.unique_merchant.count)',
    'user:history.offline_buy_with_coupon.discount_rate.min',
    'user_merchant:longago(offline_buy_with_coupon.count/offline_receive_coupon.count)',
    'user:longago.offline_buy_with_coupon.discount_rate.max',
    'merchant:longago.offline_buy_with_coupon.distance.min',
    'coupon:longago.offline_buy_with_coupon.distance.mean',
    'user:recent(offline_buy_with_coupon.unique_merchant.count/offline.unique_merchant.count)',
    'coupon:recent.offline_buy_with_coupon.distance.max',
    'coupon:longago.offline_buy_with_coupon.timedelta.mean',
    'user_merchant:history(offline_buy_with_coupon.count/offline_receive_coupon.count)',
    'user:history.offline_buy_with_coupon.discount_rate.max',
    'user:history.online_buy_with_coupon.count',
    'user_merchant:longago.offline_receive_coupon.count',
    'coupon:history.offline_buy_with_coupon.unique_user.count',
    'merchant:longago.offline.unique_hotcoupon.count',
    'user:history(offline_buy_with_coupon.unique_merchant.count/offline.unique_merchant.count)',
    'user:longago(offline_buy_with_coupon.man200.count/offline_receive_coupon.man200.count)',
    'merchant:history.offline_buy_with_coupon.distance.min',
    'merchant:longago.offline.unique_coupon.count',
    'coupon:recent.offline_receive_coupon.distance.min',
    'user:longago(offline_buy_with_coupon.unique_merchant.count/offline.unique_merchant.count)',
    'merchant:recent.offline.unique_coupon.count',
    'user:history.offline_buy_with_coupon.unique_merchant.count',
    'user:history.offline_buy_with_coupon.distance.max',
    'user:recent.offline_buy_with_coupon.distance.min',
    'coupon:longago.offline_buy_with_coupon.unique_user.count',
    'user:longago.offline_buy_with_coupon.count',
    'merchant:history.offline.unique_hotcoupon.count',
    'user:recent.offline_buy_with_coupon.distance.max',
    'user:recent.offline_buy_with_coupon.distance.mean',
    'user:history.offline_buy_with_coupon.man200.count',
    'user:longago.offline_buy_with_coupon.distance.max',
    'user:history.offline_buy_with_coupon.distance.min',
    'user:history.offline_buy_with_coupon.distance.mean',
    'user:longago.offline_buy_with_coupon.distance.mean',
    'merchant:recent.offline.unique_hotcoupon.count',
    'coupon:longago.offline_buy_with_coupon.count',
    'merchant:history.offline.unique_coupon.count',
    'user:recent(offline_buy_with_coupon.unique_coupon.count/offline.unique_coupon.count)',
    'coupon:history.offline_buy_with_coupon.distance.mean',
    'coupon:recent.offline_buy_with_coupon.distance.min',
    'coupon:longago.offline_buy_with_coupon.distance.max',
    'coupon:history.offline_receive_coupon.distance.max',
    'coupon:longago.offline_receive_coupon.distance.max',
    'user:longago.offline_buy_with_coupon.unique_merchant.count',
    'coupon:history.offline_buy_with_coupon.distance.min',
    'user:recent.offline_buy_with_coupon.unique_merchant.count',
    'user:history.offline_buy_with_coupon.unique_coupon.count',
    'user_merchant:longago.offline_buy_with_coupon.count',
    'coupon:history.offline_buy_with_coupon.distance.max',
    'user:recent.offline_buy_with_coupon.unique_coupon.count',
    'user:longago.offline_buy_with_coupon.distance.min',
    'coupon:longago.offline_receive_coupon.distance.min',
    'user:longago(offline_buy_with_coupon.unique_coupon.count/offline.unique_coupon.count)',
    'user:history(offline_buy_with_coupon.unique_coupon.count/offline.unique_coupon.count)',
    'coupon:longago.offline_buy_with_coupon.distance.min',
    'user:longago.offline_buy_with_coupon.unique_coupon.count',
    'user:longago(offline_buy_with_coupon.man200.count/offline_buy_with_coupon.count)',
    'coupon:history.offline_receive_coupon.distance.min',
    'user:history(offline_buy_with_coupon.man200.count/offline_receive_coupon.man200.count)',
    'user:longago.offline_buy_with_coupon.man200.count',
    'user:recent(offline_buy_with_coupon.man200.count/offline_buy_with_coupon.count)',
    'user:recent.offline_buy_with_coupon.man200.count',
    'user:history(offline_buy_with_coupon.man200.count/offline_buy_with_coupon.count)',
    'user:recent(offline_buy_with_coupon.man200.count/offline_receive_coupon.man200.count)',
    'is_xianshi',
    'merchant:today.offline_receive_coupon.unique_coupon.count'
]


def read_dataset():
    df_test = pd.read_msgpack('data/z6_ts_merged_test.msgpack')
    df_test_full = pd.read_msgpack('data/z6_ts_merged_test_full.msgpack')
    features = list(df_test.columns.difference(['date', 'label']))
    features = [x for x in features if 'future' not in x and 'today' not in x]
    # features = [x for x in features if ':' not in x]
    # features = [x for x in features if 'offline' not in x]
    # features = [x for x in features if 'user:' not in x]
    # features = [x for x in features if 'merchant:' not in x]
    # features = [x for x in features if 'coupon:' not in x]
    # features = [x for x in features if 'user_merchant:' not in x]
    user_features = [x for x in FEATURE_TOP[:200] if 'user:' in x]
    features = list(sorted(set(user_features) | set(FEATURE_TOP[:80])))
    print(pretty(features))
    df_test = df_test[features]
    cols = ['user_id', 'coupon_id', 'date']
    if TEST_HAS_LABEL:
        cols += ['label']
    df_submit = df_test_full[cols].copy()
    df_submit['date'] = df_submit.date.dt.strftime('%Y%m%d')
    LOG.info('df_test {}', df_test.shape)
    LOG.info('df_submit {}', df_submit.shape)

    submit_cols = ['user_id', 'coupon_id', 'date', 'label']
    df_train = pd.read_msgpack('data/z6_ts_merged_train.msgpack')
    df_train_full = pd.read_msgpack('data/z6_ts_merged_train_full.msgpack')
    mask = np.random.rand(len(df_train)) < 0.05

    df_validate = df_train.loc[mask, features + ['label']]
    df_validate_submit = df_train_full.loc[mask, submit_cols].copy()
    df_validate_submit['date'] = df_validate_submit.date.dt.strftime('%Y%m%d')

    df_train = df_train.loc[~mask, features + ['label']]
    df_train_submit = df_train_full.loc[~mask, submit_cols].copy()
    df_train_submit['date'] = df_train_submit.date.dt.strftime('%Y%m%d')
    LOG.info('df_train {}', df_train.shape)
    LOG.info('df_train_submit {}', df_train_submit.shape)
    LOG.info('df_validate {}', df_validate.shape)
    LOG.info('df_validate_submit {}', df_validate_submit.shape)

    df_train_x, df_train_y = split_feature_label(df_train)
    df_validate_x, df_validate_y = split_feature_label(df_validate)

    return df_train_x, df_train_y, df_validate_x, df_validate_y, df_test, df_submit, df_validate_submit, df_train_submit


(
    df_train_x_best,
    df_train_y,
    df_validate_x_best,
    df_validate_y,
    df_test_x_best,
    df_submit,
    df_validate_submit,
    df_train_submit,
) = read_dataset()
SUBMIT_NAME = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')


def lgb_predict():
    params = {
        # 'boosting': 'dart',
        'objective': 'binary',
        'metric': 'auc',
        # 'is_unbalance': True,
        'max_depth': 12,
        'max_leaves': 16,
        # 'max_bin': 63,
        # 'min_data_in_leaf': 50,
        'feature_fraction': 0.2,
        'scale_pos_weight': 2,
        # 'lambda_l1': 10,
        # 'lambda_l2': 0.1,
        'eta': 0.01,
        # 'zero_as_missing': True,
        'num_threads': 6,
        'num_trees': 5000,
        'early_stopping_round': 100,
    }
    dataset_train = lgb.Dataset(df_train_x_best, label=df_train_y)
    dataset_validate = lgb.Dataset(df_validate_x_best, label=df_validate_y)
    valid_sets = [dataset_train, dataset_validate]
    valid_names = ['train', 'validate']

    LOG.info('train begin')
    model = lgb.train(params, dataset_train, verbose_eval=20,
                      valid_sets=valid_sets, valid_names=valid_names)
    LOG.info('train end')
    model.save_model(f'data/model-lgb-{SUBMIT_NAME}.dat')

    test_label = model.predict(df_test_x_best)
    validate_label = model.predict(df_validate_x_best)
    train_label = model.predict(df_train_x_best)
    return test_label, validate_label, train_label


def xgb_predict():
    params = {
        'booster': 'gbtree',
        'objective': 'rank:pairwise',
        'eval_metric': 'auc',
        'gamma': 0.1,
        'min_child_weight': 1.1,
        'max_depth': 12,
        'max_leaves': 128,
        'lambda': 10,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'colsample_bylevel': 0.7,
        'eta': 0.01,
        'tree_method': 'exact',
        'seed': 0,
        'nthread': 4,
        'verbosity': 0,
        'metric_freq': 100,
    }
    dataset_train = xgb.DMatrix(df_train_x_best, label=df_train_y)
    dataset_validate = xgb.DMatrix(df_validate_x_best, label=df_validate_y)
    dataset_test = xgb.DMatrix(df_test_x_best)

    watchlist = [(dataset_train, 'train'), (dataset_validate, 'validate')]

    LOG.info('train begin')
    model = xgb.train(params, dataset_train, num_boost_round=500, evals=watchlist)
    LOG.info('train end')
    model.save_model(f'data/model-xgb-{SUBMIT_NAME}.dat')

    predict_label = model.predict(dataset_test)
    validate_label = model.predict(dataset_validate)
    train_label = model.predict(dataset_train)
    return predict_label, validate_label, train_label


def main():
    use_xgb = len(sys.argv) >= 2 and sys.argv[1] == 'xgb'
    if use_xgb:
        submit_path = f'data/submit-xgb-{SUBMIT_NAME}.csv'
        validate_submit_path = f'data/submit-xgb-validate-{SUBMIT_NAME}.csv'
        train_submit_path = f'data/submit-xgb-train-{SUBMIT_NAME}.csv'
        test_label, validate_label, train_label = xgb_predict()
    else:
        submit_path = f'data/submit-lgb-{SUBMIT_NAME}.csv'
        validate_submit_path = f'data/submit-lgb-validate-{SUBMIT_NAME}.csv'
        train_submit_path = f'data/submit-lgb-train-{SUBMIT_NAME}.csv'
        test_label, validate_label, train_label = lgb_predict()

    df_submit['prob'] = test_label
    LOG.info('save result {}', submit_path)
    df_submit.to_csv(submit_path, index=False, header=False)
    if TEST_HAS_LABEL:
        print('test o2o auc: {:.3f}'.format(o2o_auc_score(df_submit)))

    df_train_submit['prob'] = train_label
    LOG.info('save train result {}', train_submit_path)
    df_train_submit.to_csv(train_submit_path, index=False, header=False)
    print('train o2o auc: {:.3f}'.format(o2o_auc_score(df_train_submit)))

    df_validate_submit['prob'] = validate_label
    LOG.info('save validate result {}', validate_submit_path)
    df_validate_submit.to_csv(validate_submit_path, index=False, header=False)
    print('validate o2o auc: {:.3f}'.format(o2o_auc_score(df_validate_submit)))


if __name__ == "__main__":
    main()
