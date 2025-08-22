# -*- coding: utf-8 -*-
"""
Created on Fri Aug  1 14:30:31 2025

@author: lch99
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ----------------------
# 1. 生成模拟数据
# ----------------------
np.random.seed(42)
years = np.arange(1990, 2021)
n = len(years)

data = pd.DataFrame({
    'year': years,
    'nuclear_plants': np.linspace(100, 150, n).astype(int) + np.random.randint(-5, 5, n),
    'investment': np.linspace(20, 50, n) + np.random.normal(0, 3, n),
    'policy_support': np.linspace(3, 7, n) + np.random.normal(0, 1, n),
    'carbon_price': np.linspace(10, 40, n) + np.random.normal(0, 5, n)
})

# 模拟目标变量（核发电量）：与特征正相关
data['nuclear_generation'] = (
    0.5 * data['nuclear_plants'] +
    1.2 * data['investment'] +
    3.0 * data['policy_support'] +
    0.8 * data['carbon_price'] +
    0.02 * (data['year'] - 1990) +
    np.random.normal(0, 5, n)  # 噪声
)

# ----------------------
# 2. 数据拆分
# ----------------------
X = data.drop(['year', 'nuclear_generation'], axis=1)  # 特征（排除年份，可用作时间轴）
y = data['nuclear_generation']  # 目标变量
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ----------------------
# 3. 模型训练与评估
# ----------------------
# 线性回归
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# 决策树回归
dt = DecisionTreeRegressor(max_depth=5, random_state=42)  # 限制深度避免过拟合
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

# 评估指标
print("线性回归：")
print(f"均方误差（MSE）：{mean_squared_error(y_test, y_pred_lr):.2f}")
print(f"决定系数（R²）：{r2_score(y_test, y_pred_lr):.2f}\n")

print("决策树回归：")
print(f"均方误差（MSE）：{mean_squared_error(y_test, y_pred_dt):.2f}")
print(f"决定系数（R²）：{r2_score(y_test, y_pred_dt):.2f}")

# ----------------------
# 4. 预测未来（示例：2021-2030年）
# ----------------------
future_years = np.arange(2021, 2031)
future_data = pd.DataFrame({
    'nuclear_plants': np.linspace(150, 180, 10).astype(int),  # 假设核电站数量增长
    'investment': np.linspace(50, 80, 10),  # 投资增加
    'policy_support': np.linspace(7, 9, 10),  # 政策支持加强
    'carbon_price': np.linspace(40, 60, 10)  # 碳价上涨
})

# 预测未来发电量
future_pred_lr = lr.predict(future_data)
future_pred_dt = dt.predict(future_data)

# ----------------------
# 5. 可视化结果
# ----------------------
plt.figure(figsize=(12, 6))
plt.plot(data['year'], data['nuclear_generation'], 'b-', label='历史数据')
plt.plot(future_years, future_pred_lr, 'r--', label='线性回归预测')
plt.plot(future_years, future_pred_dt, 'g--', label='决策树预测')
plt.xlabel('年份')
plt.ylabel('核能源发电量（太瓦时）')
plt.title('核能源应用发展预测')
plt.legend()
plt.grid()
plt.show()