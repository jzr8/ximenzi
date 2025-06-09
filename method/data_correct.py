

# 利用偏差去修正预测 得到的结果
def stat_correct(st_list, get_stand):
    # 符合相应指标的炉子数量
    stat_accord = {'Fe': 0, 'Cl': 0, 'C': 0, 'N': 0, 'O': 0, 'Ni': 0, 'Cr': 0, 'hard': 0, 'stand': 0, 'grade': 0}
    # 相应指标的偏差总和
    stat_error_sum = {'Fe': 0, 'Cl': 0, 'C': 0, 'N': 0, 'O': 0, 'Ni': 0, 'Cr': 0, 'hard': 0}
    # 相应指标的平均偏差
    stat_error_ave = {'Fe': 0, 'Cl': 0, 'C': 0, 'N': 0, 'O': 0, 'Ni': 0, 'Cr': 0, 'hard': 0}
    # 相应指标的偏差最大值  这里找到最大的10炉数据 按照升序排序 0索引代表最小的数据  9索引代表最大的数据
    stat_error_max = {'Fe': [0] * 10, 'Cl': [0] * 10, 'C': [0] * 10, 'N': [0] * 10, 'O': [0] * 10, 'Ni': [0] * 10,
                      'Cr': [0] * 10, 'hard': [0] * 10}
    stat_Stove_item_max = {'Fe': [None] * 10, 'Cl': [None] * 10, 'C': [None] * 10, 'N': [None] * 10, 'O': [None] * 10,
                           'Ni': [None] * 10, 'Cr': [None] * 10, 'hard': [None] * 10}

    st_num = len(st_list)  # 得到总炉数
    # 遍历每个炉数据
    for st in st_list:
        st.judge_error_correct(get_stand)  # 根据设定的标准偏差进行偏差判断

        # 符合标准的数量 & 符合等级的数量
        if -1 not in [st.Fe[0], st.Cl[0], st.C[0], st.N[0], st.O[0], st.Ni[0], st.Cr[0], st.hard[0]]:
            stat_accord['stand'] += 1
        if st.grade[0] == st.grade[1]:
            stat_accord['grade'] += 1

        # 各个指标符合的数量
        for key in stat_accord:
            if key != 'stand' and key != 'grade':
                stat_accord[key] += 1 if getattr(st, key)[0] == 1 else 0

        # 各个指标的偏差总和
        for key in stat_error_sum:
            stat_error_sum[key] += abs(getattr(st, key)[4] - getattr(st, key)[5])

        # 各个指标的偏差最大值  对该炉的数据 进行判断排序
        for key in stat_error_max:
            if abs(getattr(st, key)[4] - getattr(st, key)[5]) >= stat_error_max[key][0]:
                # 添加到列表中
                stat_error_max[key].append(abs(getattr(st, key)[4] - getattr(st, key)[5]))
                stat_Stove_item_max[key].append(st)
                combined = list(zip(stat_error_max[key], stat_Stove_item_max[key]))  # 将两个列表打包为一对一元组列表
                combined.sort(key=lambda x: x[0])  # 按数值升序排序
                combined.pop(0)  # 删除最小值（排在第一位的元素）
                new_nums, new_labels = zip(*combined)  # 解包为两个新列表
                # 转换为列表类型（因为 zip 返回的是元组）
                stat_error_max[key] = list(new_nums)
                stat_Stove_item_max[key] = list(new_labels)

    # # 找到最大值的生产日期和批次
    # for key in stat_Stove_item_max:
    #     print(key+f':date:{stat_Stove_item_max[key].date}--num:{stat_Stove_item_max[key].number}')

    # 找到偏差最大的那几个炉子的等级
    # for key in stat_Stove_item_max:
    #     print(key + f':{[stat_Stove_item_max[key][i].grade[1] for i in range(len(stat_Stove_item_max[key]))]}')

    # 计算各指标偏差的平均值
    for key in stat_error_ave:
        stat_error_ave[key] = stat_error_sum[key] / st_num

    # 符合率
    keys_to_extract = ['Fe', 'Cl', 'C', 'N', 'O', 'Ni', 'Cr', 'hard']
    stat_accord_p = {k: stat_accord[k] / st_num for k in keys_to_extract if k in stat_accord}
    # 偏差最大值
    data_max = [stat_error_max[key][-1] for key in stat_error_max]
    data_max = [round(num, 3) for num in data_max]  # 保留小数点后三位
    # 平均偏差
    data_error_ave = [stat_error_ave[key] for key in stat_error_ave]
    data_error_ave = [round(num, 3) for num in data_error_ave]  # 保留小数点后三位

    # print(st_num)

    # 平均偏差  最大偏差  符合数量  符合率  标准符合率  等级符合率
    return data_error_ave, data_max, stat_accord, stat_accord_p, stat_accord['stand'] / st_num, stat_accord[
        'grade'] / st_num