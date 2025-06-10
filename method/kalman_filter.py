from filterpy.kalman import KalmanFilter
import numpy as np
from collections import defaultdict
from .data_correct import *


#  kalman滤波器初始化
def kalman_init(x0):
    # 初始化一维 Kalman 滤波器
    kf = KalmanFilter(dim_x=2, dim_z=1)
    # 状态转移矩阵：x = [位置, 速度]
    kf.F = np.array([[1, 1],
                     [0, 1]])
    # 观测矩阵：我们只能观测位置
    kf.H = np.array([[1, 0]])
    # 初始状态估计
    kf.x = np.array([[x0], [0]])  # 初始位置和速度
    # 状态协方差矩阵
    kf.P *= 1000.
    # 过程噪声协方差
    kf.Q = np.array([[1, 0],
                     [0, 1]])
    # 观测噪声协方差
    kf.R = np.array([[0.1]])
    return kf


# 按天进行卡尔曼预测
# 返回下一天的预测偏差，过程中的所有预测偏差（包括第一个数据的预测到下一天的预测）
def kalman_by_date(ST, name, gate):
    # 按日期分组
    value_dict = defaultdict(list)
    for s in ST:
        getattr(s, name)[5] = 0  # 清0
        value_dict[s.date].append(s)

    # 排序日期（按月份和日）
    def parse_date(s):
        return tuple(map(int, s.split('_')))

    # 得到排序后的 字典的 键   即日期
    sorted_dates = sorted(value_dict.keys(), key=parse_date)

    ave_error = []  # 保存各天的平均偏差
    pre_error = None  # 对下一次的预测偏差
    pre_error_list = []  # 保存所有的预测偏差
    kf = None  # 卡尔曼滤波器

    # 按日期顺序
    for d in sorted_dates:
        e_sum = 0
        e_num = 0
        for st in value_dict[d]:
            # 对每个炉子的对应元素偏差 基于设定的阈值 进行筛选
            if getattr(st, name)[3] < gate:
                e_sum += getattr(st, name)[4]
                e_num += 1

            # 预测了当天的偏差 则赋予相应预测值
            if pre_error is not None:
                getattr(st, name)[5] = pre_error

        if e_num != 0:
            ave_error.append(e_sum / e_num)  # 计算一天的平均偏差
        #  如果该天的数据无效，则利用预测值代替实际值
        elif pre_error is not None:
            ave_error.append(pre_error)

        # 对于第一天的炉数据，进行初始化
        if (pre_error is None) and (len(ave_error) == 1):
            kf = kalman_init(ave_error[0])
            kf.predict()
            pre_error_list.append(ave_error[0])  # 第一个无法预测，直接用实际偏差代替

        if kf is not None:
            kf.update(ave_error[-1])
            kf.predict()
            pre_error = kf.x[0, 0]
            pre_error_list.append(pre_error)

    return pre_error, pre_error_list


# 按炉次进行卡尔曼预测
# 返回下一炉的预测偏差，过程中的所有预测偏差（包括第一个数据的预测到下一炉的预测）
def kalman_by_num(ST, name, gate):
    # 排序  按照日期从小到大  序号从小到大
    ST.sort(key=lambda s: (s.parse_date(), s.number))
    for s in ST:
        getattr(s, name)[5] = 0  # 清0

    ave_error = []  # 保存各炉的平均偏差
    pre_error = None  # 对下一次的预测偏差
    pre_error_list = []  # 保存所有的预测偏差
    kf = None  # 卡尔曼滤波器

    for st in ST:
        # 对每个炉子的对应元素偏差 基于设定的阈值 进行筛选
        if getattr(st, name)[3] < gate:
            ave_error.append(getattr(st, name)[4])
        # 若该炉数据无效，则用预测值代替
        elif pre_error is not None:
            ave_error.append(pre_error)

        # 预测了当天的偏差 则赋予相应预测值
        if pre_error is not None:
            getattr(st, name)[5] = pre_error

        # 对于第一天的炉数据，进行初始化
        if (pre_error is None) and (len(ave_error) == 1):
            kf = kalman_init(ave_error[0])
            kf.predict()
            pre_error_list.append(ave_error[0])  # 第一个无法预测，直接用实际偏差代替

        if kf is not None:
            kf.update(ave_error[-1])
            kf.predict()
            pre_error = kf.x[0, 0]
            pre_error_list.append(pre_error)

    return pre_error, pre_error_list


# 设置指定的阈值（gate）、权重（theta），返回指定元素的符合率（通过EWMA修正后）
def kalman_correct(ST, stand, name, gate):
    next_pre_error, pre_error_list = kalman_by_num(ST, name, gate)
    # 进行修正的估计 后的数据统计
    d_error_ave, d_max, acc, acc_p, st_per, gr_per = stat_correct(ST, stand)
    print(f'使用kalman预测偏差，得到新的符合率：{acc_p[name]*100}%')
    draw_error_by_number(ST, name, pre_error_list)
