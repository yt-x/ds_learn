# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 22:26:20 2025

@author: lch99
"""

import pandas as pd

# 模拟用户购买数据
data = {
    '用户ID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    '性别': ['男', '女', '男', '女', '男', '女', '男', '女', '男', '女'],
    '购买产品': ['A', 'B', 'A', 'A', 'B', 'B', 'A', 'B', 'A', 'B'],
    '购买次数': [2, 1, 3, 1, 2, 3, 4, 2, 1, 3]
}
df = pd.DataFrame(data)
print(df)
# 1. 基础交叉表：统计不同性别购买各产品的用户数（默认计数）
crosstab1 = pd.crosstab(
    index=df['性别'],    # 行：性别
    columns=df['购买产品']  # 列：购买产品
)
print("1. 性别×产品的用户数量分布：")
print(crosstab1, "\n")

# 2. 带百分比的交叉表：按行/列计算占比
crosstab2 = pd.crosstab(
    df['性别'],
    df['购买产品'],
    normalize='index'  # 按行计算百分比（'columns'按列，'all'总百分比）
)
print("2. 各性别购买产品的比例：")
print(crosstab2.round(2), "\n")

# 3. 基于数值的交叉表：计算购买次数总和
crosstab3 = pd.crosstab(
    df['性别'],
    df['购买产品'],
    values=df['购买次数'],  # 基于购买次数计算
    aggfunc='sum'          # 聚合方式：求和
)
print("3. 性别×产品的总购买次数：")
print(crosstab3)
