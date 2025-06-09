from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from .data_correct import *


# 滑动窗口线性回归
# 对列表s的后window_size个数据进行线性回归
# 返回下一个的预测值
def sliding_window_lr(s, window_size):
    s = np.asarray(s)
    n = len(s)
    x = np.arange(window_size).reshape(-1, 1)  # 横坐标

    y = s[-window_size:]
    model = LinearRegression()
    model.fit(x, y)
    pre_next_data = model.predict([[window_size]])  # 预测下一个值

    return pre_next_data[0]


# 为ST列表中的每个炉子 基于滑动窗口线性回归 进行偏差预测
# 同时返回下一天炉子的偏差预测值（以天为单位进行预测）
def wlr_by_date(ST, name, gate, window_size):
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
        # 若当天的数据全部无效，则用预测的偏差作为该天的平均偏差
        else:
            if pre_error is not None:
                ave_error.append(pre_error)

        # 超过窗口长度 开始正常预测
        if len(ave_error) >= window_size:
            pre_error = sliding_window_lr(ave_error, window_size)
        # 如果列表不为空，将列表中的数据全部拿来进行线性回归（数量肯定少于window_size）
        # 作为下一天的预测偏差
        elif ave_error:
            pre_error = sliding_window_lr(ave_error, len(ave_error))

    return pre_error


# 为ST列表中的每个炉子 基于滑动窗口线性回归 进行偏差预测
# 同时返回下一个炉子的偏差预测值（以炉为单位进行预测）
def wlr_by_num(ST, name, gate, window_size):
    # 排序  按照日期从小到大  序号从小到大
    ST.sort(key=lambda s: (s.parse_date(), s.number))
    for s in ST:
        getattr(s, name)[5] = 0  # 清0

    ave_error = []  # 保存各炉的平均偏差
    pre_error = None  # 对下一次的预测偏差

    for st in ST:
        # 对每个炉子的对应元素偏差 基于设定的阈值 进行筛选
        if getattr(st, name)[3] < gate:
            ave_error.append(getattr(st, name)[4])
        # 若该炉数据无效，则用预测值代替
        elif pre_error is not None:
            ave_error.append(pre_error)

        if pre_error is not None:
            getattr(st, name)[5] = pre_error

        # 超过窗口长度 开始正常预测
        if len(ave_error) >= window_size:
            pre_error = sliding_window_lr(ave_error, window_size)
        # 如果列表不为空，将列表中的数据全部拿来进行线性回归（数量肯定少于window_size）
        # 作为下一炉的预测偏差
        elif ave_error:
            pre_error = sliding_window_lr(ave_error, len(ave_error))

    return pre_error


def opti_window_size(ST, stand, name):
    max_size = 15
    window_size_arange = np.arange(1, max_size + 1)  # 从1到max_size
    v = []

    for window_size in window_size_arange:
        pre_error = wlr_by_num(ST, name, gate=0.009, window_size=window_size)
        # 进行修正的估计 后的数据统计
        d_error_ave, d_max, acc, acc_p, st_per, gr_per = stat_correct(ST, stand)
        v.append(acc_p[name])

    print(f'最高符合率:{max(v)}, 对应的window_size:{window_size_arange[v.index(max(v))]}')
    plt.plot(window_size_arange, v, marker='o')  # marker='o' 显示点
    plt.xlabel('window_size')
    plt.ylabel(name)
    plt.title('window_size-percent')
    plt.grid(True)
    plt.show()


# 设置指定的阈值（gate）、权重（theta），返回指定元素的符合率（通过EWMA修正后）
def WLR_correct(ST, stand, name, gate, window_size):
    next_pre_error = wlr_by_num(ST, name, gate, window_size)
    # 进行修正的估计 后的数据统计
    d_error_ave, d_max, acc, acc_p, st_per, gr_per = stat_correct(ST, stand)
    print(f'使用window_LR预测偏差，得到新的符合率：{acc_p[name]*100}%')
