import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import math
from .data_correct import *


# 输入历史的炉子信息，为过去的每个炉子赋予加权指数偏差，同时返回下一天炉子的估计偏差
# 以及预测过程中的所有预测偏差（包括第一个数据的预测到下一天的预测）
def correct_predict_by_date(ST, name, gate, theta):
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
    pre_error_list = []

    # 按日期顺序 计算指数加权偏差
    for d in sorted_dates:
        e_sum = 0
        e_num = 0
        for st in value_dict[d]:
            # 对每个炉子的对应元素偏差 基于设定的阈值 进行筛选
            if getattr(st, name)[3] < gate:
                e_sum += getattr(st, name)[4]
                e_num += 1
            # 限幅操作
            # else:
            #     if getattr(st, name)[4] > 0:
            #         e_sum += gate
            #     else:
            #         e_sum += -gate
            #     e_num += 1

            # 对每个炉子 添加指数加权偏差（除了第一个炉子外）
            if correct_error is not None:
                getattr(st, name)[5] = correct_error

        if correct_error is not None:
            pre_error_list.append(correct_error)

        if correct_error is None:
            if e_num != 0:
                correct_error = e_sum / e_num
                pre_error_list.append(correct_error)
        else:
            if e_num != 0:
                correct_error = theta * (e_sum / e_num) + (1 - theta) * correct_error

    pre_error_list.append(correct_error)  # 加上最后的预测
    return correct_error, pre_error_list


# 输入历史的炉子信息，为过去的每个炉子赋予加权指数偏差，同时返回下一个炉子的估计偏差
# 以及预测过程中的所有预测偏差（包括第一个数据的预测到下一炉的预测）
def correct_predict_by_num(ST, name, gate, theta):
    # 排序  按照日期从小到大  序号从小到大
    ST.sort(key=lambda s: (s.parse_date(), s.number))
    for s in ST:
        getattr(s, name)[5] = 0  # 清0

    correct_error = None
    pre_error_list = []

    for st in ST:
        # 对每个炉子 添加指数加权偏差（除了第一个炉子外）
        if correct_error is not None:
            getattr(st, name)[5] = correct_error
            pre_error_list.append(correct_error)
        # 对每个炉子的对应元素偏差 基于设定的阈值 进行筛选
        if getattr(st, name)[3] < gate:
            if correct_error is None:
                correct_error = getattr(st, name)[4]
                pre_error_list.append(correct_error)
            else:
                correct_error = theta * getattr(st, name)[4] + (1 - theta) * correct_error
        # 限幅操作
        # else:
        #     if correct_error is None:
        #         if getattr(st, name)[4] > 0: correct_error = gate
        #         else: correct_error = -gate
        #     else:
        #         if getattr(st, name)[4] > 0:
        #             correct_error = theta * gate + (1 - theta) * correct_error
        #         else:
        #             correct_error = theta * (-gate) + (1 - theta) * correct_error

    pre_error_list.append(correct_error)  # 加上最后的预测
    return correct_error, pre_error_list


# 将一个参数固定  得到另一个参数关于符合率的变化曲线图
def correct_opti_one(ST, stand, name):
    # 设置寻找哪个参数：theta/gate
    method = 'theta'
    # method = 'gate'

    theta_arange = np.linspace(0, 1, 100)  # 从0到1，等间隔分成100个点（包含1）
    gate_arange = np.linspace(0, 0.074, 100)
    v = []

    # 寻找最优theta
    if method == 'theta':
        for theta in theta_arange:
            next_pre_error = correct_predict_by_num(ST, name, 0.05, theta)
            # 进行修正的估计 后的数据统计
            d_error_ave, d_max, acc, acc_p, st_per, gr_per = stat_correct(ST, stand)
            v.append(acc_p[name])
    # 寻找最优gate
    elif method == 'gate':
        for gate in gate_arange:
            next_pre_error = correct_predict_by_num(ST, name, gate, 0.0707)
            # 进行修正的估计 后的数据统计
            d_error_ave, d_max, acc, acc_p, st_per, gr_per = stat_correct(ST, stand)
            v.append(acc_p[name])
    else:
        print('method is not exist')

    # 绘图
    if method == 'theta':
        print(f'最高符合率:{max(v)}, 对应的theta:{theta_arange[v.index(max(v))]}')
        plt.plot(theta_arange, v, marker='o')  # marker='o' 显示点
    else:
        print(f'最高符合率:{max(v)}, 对应的gate:{gate_arange[v.index(max(v))]}')
        plt.plot(gate_arange, v, marker='o')  # marker='o' 显示点
    plt.xlabel('value')
    plt.ylabel(name)
    plt.title('value-percent')
    plt.grid(True)
    plt.show()


# 寻找最优的theta和gate搭配
def correct_opti_two(ST, stand, name):
    theta_arange = np.linspace(0, 1, 100)  # 从0到1，等间隔分成100个点（包含1）
    gate_arange = np.linspace(0, 0.054, 100)
    max_percent = 0
    opt_theta = 0
    opt_gate = 0
    for theta in theta_arange:
        for gate in gate_arange:
            next_pre_error = correct_predict_by_num(ST, name, gate, theta)
            # 进行修正的估计 后的数据统计
            d_error_ave, d_max, acc, acc_p, st_per, gr_per = stat_correct(ST, stand)
            if acc_p[name] > max_percent:
                max_percent = acc_p[name]
                opt_theta = theta
                opt_gate = gate
    print(f'最优gate:{opt_gate}, 最优theta:{opt_theta}, 最高符合率:{max_percent}')


# 设置指定的阈值（gate）、权重（theta），返回指定元素的符合率（通过EWMA修正后）
def EWMA_correct(ST, stand, name, gate, theta):
    next_pre_error, pre_error_list = correct_predict_by_num(ST, name, gate, theta)
    # 进行修正的估计 后的数据统计
    d_error_ave, d_max, acc, acc_p, st_per, gr_per = stat_correct(ST, stand)
    print(f'使用EWMA预测偏差，得到新的符合率：{acc_p[name]*100}%')
    draw_error_by_number(ST, name, pre_error_list)
