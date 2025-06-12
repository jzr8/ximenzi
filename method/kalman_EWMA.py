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
    kf.P *= 10.
    # 过程噪声协方差
    kf.Q = np.array([[1, 0],
                     [0, 1]])
    # 观测噪声协方差
    kf.R = np.array([[0.1]])
    return kf


# 按天预测：卡尔曼预测的结果 与 EWMA预测的结果 的平均预测 作为最终预测值
# 返回下一天的预测偏差，以及过程中所有的预测偏差
# alpha为EWMA的权重  1-alpha为kalman的权重   最终预测偏差为 alpha*EWMA + (1-alpha)*kalman
def kalman_sum_EWMA_by_date(ST, name, gate, theta, alpha):
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

    correct_error = None
    ave_error = []  # 保存各天的平均偏差
    pre_error = None  # 对下一次的预测偏差
    pre_error_list = []  # 保存所有的预测偏差
    kf = None  # 卡尔曼滤波器

    # 按日期顺序 计算指数加权偏差
    for d in sorted_dates:
        e_sum = 0
        e_num = 0
        for st in value_dict[d]:
            # 对每个炉子的对应元素偏差 基于设定的阈值 进行筛选
            if getattr(st, name)[3] < gate:
                e_sum += getattr(st, name)[4]
                e_num += 1

            # 对每个炉子 添加指数加权偏差（除了第一个炉子外）
            if (correct_error is not None) and (pre_error is not None):
                getattr(st, name)[5] = alpha * correct_error + (1-alpha) * pre_error

        if (correct_error is not None) and (pre_error is not None):
            pre_error_list.append(alpha * correct_error + (1-alpha) * pre_error)

        # 对于第一天的炉数据，进行初始化
        if (correct_error is None) and (pre_error is None):
            if e_num != 0:
                ave_error.append(e_sum / e_num)  # 计算一天的平均偏差
                kf = kalman_init(ave_error[0])
                kf.predict()
                correct_error = e_sum / e_num
                pre_error_list.append(alpha * correct_error + (1-alpha) * ave_error[0])
        else:
            if e_num != 0:
                ave_error.append(e_sum / e_num)  # 计算一天的平均偏差
                correct_error = theta * (e_sum / e_num) + (1 - theta) * correct_error
            #  如果该天的数据无效，则利用预测值代替实际值
            elif pre_error is not None:
                ave_error.append(pre_error)

        if kf is not None:
            kf.update(ave_error[-1])
            kf.predict()
            pre_error = kf.x[0, 0]

    if (correct_error is None) or (pre_error is None):
        final_error = None
    else:
        final_error = alpha * correct_error + (1-alpha) * pre_error
        pre_error_list.append(final_error)  # 加上最后的预测
    return final_error, pre_error_list


# 按炉预测：卡尔曼预测的结果 与 EWMA预测的结果 的平均预测 作为最终预测值
# 返回下一炉的预测偏差，以及过程中所有的预测偏差
# alpha为EWMA的权重  1-alpha为kalman的权重   最终预测偏差为 alpha*EWMA + (1-alpha)*kalman
def kalman_sum_EWMA_by_num(ST, name, gate, theta, alpha):
    # 排序  按照日期从小到大  序号从小到大
    ST.sort(key=lambda s: (s.parse_date(), s.number))
    for s in ST:
        getattr(s, name)[5] = 0  # 清0

    correct_error = None  # EWMA下一次的预测偏差
    ave_error = []  # 保存各炉的平均偏差
    pre_error = None  # kalman下一次的预测偏差
    pre_error_list = []  # 保存所有的预测偏差
    kf = None  # 卡尔曼滤波器

    for st in ST:
        # 对每个炉子 添加指数加权偏差（除了第一个炉子外）
        if (correct_error is not None) and (pre_error is not None):
            getattr(st, name)[5] = alpha * correct_error + (1-alpha) * pre_error
            pre_error_list.append(alpha * correct_error + (1-alpha) * pre_error)
        # 对每个炉子的对应元素偏差 基于设定的阈值 进行筛选
        if getattr(st, name)[3] < gate:
            ave_error.append(getattr(st, name)[4])
            # 对于第一天的炉数据，进行初始化
            if (correct_error is None) and (pre_error is None) and (len(ave_error) == 1):
                correct_error = getattr(st, name)[4]
                kf = kalman_init(ave_error[0])
                kf.predict()
                pre_error_list.append(alpha * correct_error + (1-alpha) * ave_error[0])
            else:
                correct_error = theta * getattr(st, name)[4] + (1 - theta) * correct_error
        # 若该炉数据无效，则用预测值代替
        elif pre_error is not None:
            ave_error.append(pre_error)

        if kf is not None:
            kf.update(ave_error[-1])
            kf.predict()
            pre_error = kf.x[0, 0]

    if (correct_error is None) or (pre_error is None):
        final_error = None
    else:
        final_error = alpha * correct_error + (1-alpha) * pre_error
        pre_error_list.append(final_error)  # 加上最后的预测
    return final_error, pre_error_list


# 寻找最优的theta和gate搭配
def kalman_EWMA_opti_two(ST, stand, name):
    theta_arange = np.linspace(0, 1, 100)  # 从0到1，等间隔分成100个点（包含1）
    gate_arange = np.linspace(0, 0.054, 100)
    max_percent = 0
    opt_theta = 0
    opt_gate = 0
    for theta in theta_arange:
        for gate in gate_arange:
            next_pre_error, _ = kalman_sum_EWMA_by_date(ST, name, gate, theta)
            # 进行修正的估计 后的数据统计
            d_error_ave, d_max, acc, acc_p, st_per, gr_per = stat_correct(ST, stand)
            if acc_p[name] > max_percent:
                max_percent = acc_p[name]
                opt_theta = theta
                opt_gate = gate
    print(f'最优gate:{opt_gate}, 最优theta:{opt_theta}, 最高符合率:{max_percent}')


# 设置指定的阈值(gate)、EWMA的权重(theta)、EWMA和kalman的权重(alpha)，返回指定元素的符合率（通过EWMA修正后）
def kalman_EWMA_correct(ST, stand, name, gate, theta, alpha):
    next_pre_error, pre_error_list = kalman_sum_EWMA_by_num(ST, name, gate, theta, alpha)
    # 进行修正的估计 后的数据统计
    d_error_ave, d_max, acc, acc_p, st_per, gr_per = stat_correct(ST, stand)
    print(f'使用kalman_EWMA预测偏差，得到新的符合率：{acc_p[name]*100}%')
    draw_error_by_number(ST, name, pre_error_list)
