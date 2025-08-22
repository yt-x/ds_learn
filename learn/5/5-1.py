# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 22:06:20 2025
@author: lch99
"""

import numpy as np
import matplotlib.pyplot as plt
# 全局设置字体（适用于所有图表）
plt.rcParams["font.family"] = ["SimHei"]  # 支持中文的字体
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 1. 生成样本数据
np.random.seed(42)
x = np.linspace(0, 10, 50)  # 输入特征
y = 2 * x + 5 + np.random.normal(0, 1, 50)  # 真实关系：y=2x+5，添加噪声

# 2. 手动实现最小二乘法
n = len(x)
x_mean = np.mean(x)
y_mean = np.mean(y)

# 计算w和b
numerator = np.sum((x - x_mean) * (y - y_mean))  # 分子：协方差之和
denominator = np.sum((x - x_mean) ** 2)          # 分母：x的方差之和
w = numerator / denominator
b = y_mean - w * x_mean

print(f"手动计算的参数：w={w:.2f}, b={b:.2f}")  # 接近真实值w=2, b=5

# 3. 用sklearn验证（库函数实现）
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x.reshape(-1, 1), y)  # 输入需为二维数组
print(f"sklearn计算的参数：w={model.coef_[0]:.2f}, b={model.intercept_:.2f}")

# 4. 可视化拟合结果
y_pred = w * x + b  # 预测值
plt.scatter(x, y, label="样本数据")
plt.plot(x, y_pred, 'r-', label=f"拟合直线：y={w:.2f}x+{b:.2f}")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

# 计算模型得分
score = model.score(x.reshape(-1,1), y)
print("模型得分:", score)
