import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# ====== 1. 构造数据 ======
data = [0.5, 0.9, -0.3, 0.2, 0.8, -0.1, 0.0, 0.6, 0.4, -0.2]
seq_len = 3  # 输入序列长度

