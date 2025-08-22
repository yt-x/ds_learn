# -*- coding: utf-8 -*-
"""
Created on Sat Jul 26 11:04:03 2025

@author: lch99
"""

import pandas as pd

# ----------------------
# 1. 准备数据（模拟不同来源的核燃料数据）
# ----------------------
# 一车间生产数据
workshop1 = pd.DataFrame({
    '燃料棒ID': ['FR-001', 'FR-002', 'FR-003'],
    '铀浓度(%)': [4.5, 4.7, 4.6],
    '生产时间': ['2023-10-01', '2023-10-02', '2023-10-03'],
    '车间': '一车间'
})

# 二车间生产数据（结构与一车间相同）
workshop2 = pd.DataFrame({
    '燃料棒ID': ['FR-004', 'FR-005'],
    '铀浓度(%)': [5.0, 4.8],
    '生产时间': ['2023-10-01', '2023-10-03'],
    '车间': '二车间'
})

# 检测数据（需与生产数据关联）
inspection = pd.DataFrame({
    '燃料棒ID': ['FR-001', 'FR-002', 'FR-003', 'FR-004', 'FR-005'],
    '检测结果': ['合格', '合格', '待复检', '合格', '不合格'],
    '检测员': ['张工', '李工', '王工', '赵工', '孙工']
})

print("原始数据：")
print("一车间数据:\n", workshop1, "\n")
print("二车间数据:\n", workshop2, "\n")
print("检测数据:\n", inspection, "\n")


# ----------------------
# 2. 使用concat()纵向合并生产数据
# ----------------------
# 合并两个车间的生产记录（纵向拼接）
production_all = pd.concat([workshop1, workshop2], ignore_index=True)
print("2. 合并后的生产数据：")
print(production_all, "\n")


# ----------------------
# 3. 使用merge()关联生产数据与检测数据
# ----------------------
# 基于"燃料棒ID"连接生产数据和检测数据（内连接）
combined_data = pd.merge(
    production_all, 
    inspection, 
    on='燃料棒ID',  # 连接键
    how='inner'     # 只保留双方都有的记录
)

print("3. 生产+检测合并数据：")
print(combined_data)
