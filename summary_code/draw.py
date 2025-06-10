from collections import defaultdict
import matplotlib.pyplot as plt


# 按日期对指定元素的偏差进行曲线绘制
def draw_error_by_date(ST, name):
    # 按日期分组
    value_dict = defaultdict(list)
    for s in ST:
        value_dict[s.date].append(getattr(s, name)[4])

    # 排序日期（按月份和日）
    def parse_date(s):
        return tuple(map(int, s.split('_')))

    # 得到排序后的 字典的 键   即日期
    sorted_dates = sorted(value_dict.keys(), key=parse_date)

    # 构造 x 和 y 轴数据
    x_labels = sorted_dates
    y_values = [sum(value_dict[d]) / len(value_dict[d]) for d in x_labels]

    # 画柱状图
    plt.figure(figsize=(8, 5))
    plt.bar(x_labels, y_values, color='skyblue')
    plt.xlabel('Date')
    plt.ylabel('Average Error')
    plt.title(f'{name} error')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)  # 旋转日期标签，防止重叠
    plt.tight_layout()
    plt.show()


# 按炉子序列号对指定元素的偏差进行曲线绘制
def draw_error_by_number(ST, name):
    # 排序  按照日期从小到大  序号从小到大
    ST.sort(key=lambda s: (s.parse_date(), s.number))

    x_labels = []
    y_values = []
    for i, st in enumerate(ST, 1):
        x_labels.append(i)
        y_values.append(st.Fe[4])

    # 画柱状图
    plt.figure(figsize=(8, 5))
    plt.bar(x_labels, y_values, color='skyblue')
    plt.xlabel('Date')
    plt.ylabel('Average Error')
    plt.title(f'{name} error')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)  # 旋转日期标签，防止重叠
    plt.tight_layout()
    plt.show()