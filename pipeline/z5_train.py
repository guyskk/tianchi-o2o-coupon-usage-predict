import os
import sys
import json
import lightgbm as lgb
import xgboost as xgb
import pandas as pd
import datetime
from loguru import logger as LOG
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.feature_selection import VarianceThreshold


def pretty(data):
    return json.dumps(data, indent=4, ensure_ascii=False)


def split_feature_label(df):
    df_x = df[df.columns.difference(['label'])]
    df_y = df['label']
    return df_x, df_y


def get_num_features(total):
    num = os.getenv('NUM_FEATURES', '').strip()
    if num == 'all':
        return num
    if num.startswith('0.'):
        num = int(total * float(num))
    elif num:
        num = int(num)
    else:
        num = int(total * 0.8)
    return num


def select_best_features(df_x, df_y):
    select = VarianceThreshold(0.01)
    select.fit(df_x, df_y)
    best_features = list(df_x.columns[select.get_support()])
    df_x = df_x[best_features]

    NUM_FEATURES = get_num_features(len(df_x.columns))

    select = SelectKBest(chi2, k=NUM_FEATURES)
    select.fit(df_x, df_y)
    best_features_chi = list(df_x.columns[select.get_support()])

    select = SelectKBest(f_classif, k=NUM_FEATURES)
    select.fit(df_x, df_y)
    best_features_clas = list(df_x.columns[select.get_support()])

    best_features = list(sorted(set(best_features_chi) & set(best_features_clas)))
    return best_features


def read_dataset():
    df_1 = pd.read_msgpack('data/z4_merge_1.msgpack')
    df_1.info()
    print('-' * 60)

    df_2 = pd.read_msgpack('data/z4_merge_2.msgpack')
    df_2.info()
    print('-' * 60)

    df_3 = pd.read_msgpack('data/z4_merge_3.msgpack')
    df_3.info()
    print('-' * 60)

    df_raw_test = pd.read_msgpack('data/z1_raw_test.msgpack')
    df_submit = df_raw_test[['user_id', 'coupon_id', 'date_received']].copy()
    df_submit['date_received'] = df_submit.date_received.dt.strftime('%Y%m%d')
    df_submit.info()
    print('-' * 60)

    df_train = pd.concat([df_1, df_2])
    # df_train = df_1
    df_validate = df_2
    df_test = df_3

    df_train_x, df_train_y = split_feature_label(df_train)
    best_features = select_best_features(df_train_x, df_train_y)
    LOG.info('best {} features:\n{}', len(best_features), pretty(best_features))

    df_train_x_best = df_train_x[best_features]

    df_validate_x, df_validate_y = split_feature_label(df_validate)
    df_validate_x_best = df_validate_x[best_features]

    df_test_x_best = df_test[best_features]

    return df_train_x_best, df_train_y, df_validate_x_best, df_validate_y, df_test_x_best, df_submit


df_train_x_best, df_train_y, df_validate_x_best, df_validate_y, df_test_x_best, df_submit = read_dataset()
SUBMIT_NAME = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')


def lgb_predict():
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'is_unbalance': True,
        'max_depth': 6,
        'max_leaves': 127,
        'lambda_l1': 0.1,
        'lambda_l2': 10,
        'eta': 0.01,
        'num_threads': 7,
    }
    dataset_train = lgb.Dataset(df_train_x_best, label=df_train_y)
    dataset_validate = lgb.Dataset(df_validate_x_best, label=df_validate_y)
    valid_sets = [dataset_train, dataset_validate]
    valid_names = ['train', 'validate']

    LOG.info('train begin')
    model = lgb.train(params, dataset_train, num_boost_round=3500,
                      valid_sets=valid_sets, valid_names=valid_names)
    LOG.info('train end')
    model.save_model(f'data/model-lgb-{SUBMIT_NAME}.dat')

    test_label = model.predict(df_test_x_best)
    return test_label


def xgb_predict():
    params = {
        'booster': 'gbtree',
        'objective': 'rank:pairwise',
        'eval_metric': 'auc',
        'gamma': 0.1,
        'min_child_weight': 1.1,
        'max_depth': 6,
        'lambda': 10,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'colsample_bylevel': 0.7,
        'eta': 0.01,
        'tree_method': 'exact',
        'seed': 0,
        'nthread': 8,
        'verbosity': 0,
    }
    dataset_train = xgb.DMatrix(df_train_x_best, label=df_train_y)
    dataset_validate = xgb.DMatrix(df_validate_x_best, label=df_validate_y)
    dataset_test = xgb.DMatrix(df_test_x_best)

    watchlist = [(dataset_train, 'train'), (dataset_validate, 'validate')]

    LOG.info('train begin')
    model = xgb.train(params, dataset_train, num_boost_round=3500, evals=watchlist)
    LOG.info('train end')
    model.save_model(f'data/model-xgb-{SUBMIT_NAME}.dat')

    predict_label = model.predict(dataset_test)
    return predict_label


def main():
    use_xgb = len(sys.argv) >= 2 and sys.argv[1] == 'xgb'
    if use_xgb:
        submit_path = f'data/submit-xgb-{SUBMIT_NAME}.csv'
        test_label = xgb_predict()
    else:
        submit_path = f'data/submit-lgb-{SUBMIT_NAME}.csv'
        test_label = lgb_predict()
    test_label_prob = MinMaxScaler()\
        .fit_transform(test_label.reshape(-1, 1))\
        .reshape(-1)
    df_submit['prob'] = test_label_prob
    LOG.info('save result {}', submit_path)
    df_submit.to_csv(submit_path, index=False, header=False)


if __name__ == "__main__":
    main()
