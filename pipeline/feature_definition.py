import numpy as np


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

USER_COUPON_FEATURE = """
future.offline_receive_coupon.count
today.offline_receive_coupon.count
future.offline_receive_coupon.deltanow.min
"""


class UserFeature:
    name = 'user'
    key_column = 'user_id'
    definition = USER_FEATURE


class MerchantFeature:
    name = 'merchant'
    key_column = 'merchant_id'
    definition = MERCHANT_FEATURE


class CouponFeature:

    name = 'coupon'
    key_column = 'coupon_id'
    definition = COUPON_FEATURE


class UserMerchantFeature:

    name = 'user_merchant'
    key_column = ['user_id', 'merchant_id']
    definition = USER_MERCHANT_FEATURE


class UserCouponFeature:
    name = 'user_coupon'
    key_column = ['user_id', 'coupon_id']
    definition = USER_COUPON_FEATURE


FEATURES = [
    UserFeature,
    MerchantFeature,
    CouponFeature,
    UserMerchantFeature,
    UserCouponFeature,
]


class ValidateSplit:
    name = 'validate'
    feature_begin = '2016-01-01'
    feature_end = '2016-04-30'
    train_begin = '2016-03-16'
    train_end = '2016-04-30'
    test_begin = '2016-05-16'
    test_end = '2016-05-31'
    test_has_label = True


class TestSplit:
    name = 'test'
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
