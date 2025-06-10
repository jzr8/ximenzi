# from sklearn.linear_model import LinearRegression
# import numpy as np
#
# y = [0.6, 0.8, 0.3, 0.9, 0.7]
# size = len(y)
# x = np.arange(size).reshape(-1, 1)  # 横坐标
# model = LinearRegression()
# model.fit(x, y)
# pre_next_data = model.predict([[size]])  # 预测下一个值
# print(pre_next_data[0])

# from filterpy.kalman import KalmanFilter
# import numpy as np

# # 你的数据列表
# s = [0.1, 0.4, 0.6, 0.8, 0.4, 0.1]
#
# # 初始化一维 Kalman 滤波器
# kf = KalmanFilter(dim_x=2, dim_z=1)
#
# # 状态转移矩阵：x = [位置, 速度]
# kf.F = np.array([[1, 1],
#                  [0, 1]])
#
# # 观测矩阵：我们只能观测位置
# kf.H = np.array([[1, 0]])
#
# # 初始状态估计
# kf.x = np.array([[s[0]], [0]])  # 初始位置和速度
#
# # 状态协方差矩阵
# kf.P *= 1.
#
# # 过程噪声协方差
# kf.Q = np.array([[1, 0],
#                  [0, 1]])
#
# # 观测噪声协方差
# kf.R = np.array([[0.1]])
#
# # 用于存储滤波后的结果
# filtered = []
#
# # 应用 Kalman 滤波
# for z in s:
#     kf.predict()
#     filtered.append(kf.x[0, 0])  # 记录当前位置估计值
#     kf.update([z])
#
# # 打印滤波结果
# print("滤波后数据：", filtered)
#
# # 预测下一个值（基于最后一个预测）
# kf.predict()
# next_prediction = kf.x[0, 0]
# print("下一个预测值：", next_prediction)
