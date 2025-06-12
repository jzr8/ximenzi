import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from .data_correct import *


# 网络训练
def train(epoch_num, lr, net, train_iter, device):
    loss = nn.MSELoss(reduction='none')
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    for epoch in range(epoch_num):
        total_loss = 0.0
        y_num = 0
        net.train()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.mean().backward()
            optimizer.step()
            with torch.no_grad():
                total_loss += l.sum()
                y_num += y.numel()
        print(f'epoch:{epoch+1}, train_loss:{total_loss/y_num:.3f}')


# 输入需要训练的偏差列表，feature_num为窗口长度（也就是特征向量的维度）
# 返回训练好的网络，以及最后的feature_num个数据（用以后续能够持续预测）
def train_get_net(error_list, feature_num):

    print(error_list[-1])

    error_torch = torch.tensor(error_list, dtype=torch.float)
    error_num = len(error_list)
    features = torch.zeros((error_num - feature_num, feature_num))  # (error_num - feature_num) * feature_num
    for i in range(feature_num):
        features[:, i] = error_torch[i: error_num - feature_num + i]
    labels = error_torch[feature_num:].reshape((-1, 1))
    final_feat = error_torch[-feature_num:]  # 用于预测下一炉偏差

    batch_size = 16
    dataset = TensorDataset(features, labels)
    train_iter = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    features = features.to(device)
    error_torch = error_torch.to(device)

    def init_weight(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    net = nn.Sequential(
        nn.Linear(feature_num, 64),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(64, 16),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(16, 4),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(4, 1),
    )
    net.apply(init_weight)
    net = net.to(device)

    train(epoch_num=100, lr=0.00001, net=net, train_iter=train_iter, device=device)

    return net, final_feat


# 设置模型超参数
def set_train_parameter(ST, name, feature_num, gate):
    # 排序  按照日期从小到大  序号从小到大
    ST.sort(key=lambda s: (s.parse_date(), s.number))
    error_list = []  # 训练的偏差数据集
    for s in ST:
        getattr(s, name)[5] = 0  # 清0
        getattr(s, name)[6] = dynamic_weight(getattr(s, name)[4], gate)
        error_list.append(getattr(s, name)[6] * 100)

    print(error_list[-1])

    error_torch = torch.tensor(error_list, dtype=torch.float)
    error_num = len(error_list)
    features = torch.zeros((error_num - feature_num, feature_num))  # (error_num - feature_num) * feature_num
    for i in range(feature_num):
        features[:, i] = error_torch[i: error_num - feature_num + i]
    labels = error_torch[feature_num:].reshape((-1, 1))

    batch_size, n_train = 16, len(ST)-100
    # 只有前n_train个样本用于训练  剩下的样本用于测试
    dataset = TensorDataset(features[:n_train], labels[:n_train])
    train_iter = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    test_features = features[n_train:]
    test_labels = labels[n_train:]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    features = features.to(device)
    error_torch = error_torch.to(device)
    test_features = test_features.to(device)
    test_labels = test_labels.to(device)

    def init_weight(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    net = nn.Sequential(
        nn.Linear(feature_num, 64),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(64, 16),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(16, 4),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(4, 1),
    )
    net.apply(init_weight)
    net = net.to(device)

    train(epoch_num=100, lr=0.00001, net=net, train_iter=train_iter, device=device)

    net.eval()
    loss = nn.MSELoss(reduction='mean')
    y_hat = net(test_features)
    l = loss(y_hat, test_labels)
    print(f'test_loss:{l:.3f}')


# 输入：ST_train训练炉数据、ST_test测试炉数据、name指定元素、feature_num窗口长度
# 按炉次进行预测 返回下一炉预测值
def mlp_by_num(ST_train, ST_test, name, feature_num, gate):
    # 排序  按照日期从小到大  序号从小到大
    ST_train.sort(key=lambda s: (s.parse_date(), s.number))
    train_error_list = []  # 训练的偏差数据集
    for s in ST_train:
        getattr(s, name)[5] = 0  # 清0
        getattr(s, name)[6] = dynamic_weight(getattr(s, name)[4], gate)
        train_error_list.append(getattr(s, name)[6] * 100)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 得到训练后的网络
    net, feature = train_get_net(train_error_list, feature_num)
    net.eval()
    feature = feature.to(device)

    # 排序  按照日期从小到大  序号从小到大
    ST_test.sort(key=lambda s: (s.parse_date(), s.number))
    for s in ST_test:
        getattr(s, name)[5] = 0  # 清0
        getattr(s, name)[6] = dynamic_weight(getattr(s, name)[4], gate)

    for st in ST_test:
        pre_error = net(feature)  # 得到当前炉次的预测偏差
        getattr(st, name)[5] = pre_error.item() / 100

        for i in range(feature_num - 1):
            feature[i] = feature[i + 1]
        feature[-1] = getattr(st, name)[6] * 100  # 将当前炉次的实际偏差加入特征向量中，以便后续预测

    next_pre_error = net(feature)

    return next_pre_error.item()


def MLP_correct(ST_train, ST_test, stand, name, feature_num, gate):
    next_pre_error = mlp_by_num(ST_train, ST_test, name, feature_num, gate)
    # 进行修正的估计 后的数据统计
    d_error_ave, d_max, acc, acc_p, st_per, gr_per = stat_correct(ST_test, stand)
    print(f'使用kalman_EWMA预测偏差，得到新的符合率：{acc_p[name]*100}%')
