# -*- coding: utf-8 -*-
"""
Created on Sun Aug  3 18:57:25 2025

@author: lch99
"""

import matplotlib.pyplot as plt
import numpy as np

data_x = ['1995', '1996', '1997', '1998', '1999', '2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009']
data_y = [0.32, 0.32, 0.32, 0.32, 0.33, 0.33, 0.34, 0.37, 0.37, 0.37, 0.37, 0.39, 0.41, 0.42, 0.44]

# 将年份字符串转换为整数类型
data_x = np.array(data_x).astype(int)

# 使用matplotlib绘制折线图
plt.figure(figsize=(8, 6))
plt.scatter(data_x, data_y, marker='o', label='Data Points')
for a,b in zip(data_x, data_y):
    plt.text(a,b,b, ha='center', va='bottom')
#help(plt.text)
# 绘制曲线
poly = np.polyfit(data_x, data_y, deg=2)#函数使用二次多项式（deg=2）对数据进行拟合，返回拟合多项式的系数。
y_value = np.polyval(poly, data_x)#函数使用拟合得到的多项式系数和原始数据点的横坐标，计算拟合曲线上对应的数据值。
plt.plot(data_x, y_value, label='Fitted Curve')

plt.title('Data Trend with Quadratic Fit')
plt.xlabel('Year')
plt.ylabel('Value')
plt.legend()
plt.show()
