from sklearn.linear_model import LinearRegression
import numpy as np

y = [0.00442, 0.00083, -0.00093, 0.00688, 0.00248]
size = len(y)
x = np.arange(size).reshape(-1, 1)  # 横坐标
model = LinearRegression()
model.fit(x, y)
pre_next_data = model.predict([[size]])  # 预测下一个值
print(pre_next_data[0])
