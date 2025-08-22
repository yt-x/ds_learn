# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 21:14:48 2025

@author: lch99
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# ----------------------
# 1. 数据准备（含重复值、缺失值、异常值）
# ----------------------
data = {
    '设备ID': ['EQ01', 'EQ02', 'EQ03', 'EQ02', 'EQ05', 'EQ06', 'EQ03', 'EQ08'],
    '温度': [85, 88, None, 88, 89, 92, None, 86],  # 含重复、缺失
    '压力': [2.1, 2.3, 2.2, 2.3, 2.5, 9.9, 2.2, 2.1],  # 含重复、异常
    '产量': [1200, 1250, 1180, 1250, 1230, 1190, 1180, 1210]
}
df = pd.DataFrame(data)
print("原始数据：")
print(df, "\n")

# ----------------------
# 2. 重复值处理
# ----------------------
# 2.1 检测重复值（完全重复的行）
duplicates = df.duplicated()
print("重复值标记（True表示重复）：")
print(duplicates.tolist(), "\n")  # [False, False, False, True, False, False, True, False]

# 2.2 查看重复行详情
print("重复行数据：")
print(df[duplicates], "\n")

# 2.3 删除重复值（保留第一次出现的行）
df = df.drop_duplicates(keep='first')
print("删除重复值后的数据：")
print(df.reset_index(drop=True), "\n")  # 重置索引

# ----------------------
# 3. 缺失值处理
# ----------------------
# 查看缺失值分布
print("缺失值统计：")
print(df.isnull().sum(), "\n")  # 仅温度列有1个缺失值
#df.isnull()
# 选择填充方法（根据数据类型和业务场景）
# 方法1：数值型数据用均值填充（适用于分布平稳的数据）
df['温度'] = df['温度'].fillna(df['温度'].mean().round(1))

print("缺失值处理后：")
print(df[['设备ID', '温度']], "\n")


# ----------------------
# 4. 异常值识别与处理
# ----------------------
def clean_outliers(df, col):
    """用IQR法识别并处理异常值"""
    # 计算四分位
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr  # 下限
    upper = q3 + 1.5 * iqr  # 上限
    
    # 识别异常值
    outliers = df[(df[col] < lower) | (df[col] > upper)][col]
    print(f"{col}异常值：{outliers.tolist()}")
    
    # 处理异常值（用上下限截断，保留数据分布）
    df[col] = df[col].clip(lower, upper)
    return df

# 处理温度和压力列的异常值
for col in ['温度', '压力']:
    df = clean_outliers(df, col)

print("\n异常值处理后：")
print(df[['设备ID', '温度', '压力']], "\n")


# ----------------------
# 5. 数据标准化
# ----------------------
# 选择需要标准化的数值列
numeric_cols = ['温度', '压力', '产量']
X = df[numeric_cols]

# 方法1：Z-score标准化（均值0，标准差1，保留分布特征）
z_scaler = StandardScaler()
df['温度_Z'] = z_scaler.fit_transform(X[['温度']])
df['压力_Z'] = z_scaler.fit_transform(X[['压力']])
df['产量_Z'] = z_scaler.fit_transform(X[['产量']])

# 方法2：Min-Max标准化（缩放到0-1，消除量纲）
mm_scaler = MinMaxScaler()
df['温度_01'] = mm_scaler.fit_transform(X[['温度']])
df['压力_01'] = mm_scaler.fit_transform(X[['压力']])
df['产量_01'] = mm_scaler.fit_transform(X[['产量']])

# 输出最终结果
print("标准化后数据（部分）：")
print(df[['设备ID', '温度', '温度_Z', '温度_01']].round(3))
