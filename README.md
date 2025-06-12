# ximenzi
# 代码结构  
## compare_before_data_load_and_rewrite_by_sheet.py  
师兄的程序，生成每个日期的表格，结果存放在**compare_out_files_by_sheet**中  
## Summary.py  
主程序  
## draw.py  
存放绘图功能的函数  
## stove.py
类
## method  
存放预测偏差的各种算法：目前有***EWMA(指数加权移动平均)***、***window_LR(滑动窗口线性回归)***、***kalman(卡尔曼滤波)***、***MLP(多层感知机)***  
