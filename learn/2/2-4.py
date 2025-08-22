# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 22:28:21 2025

@author: lch99
"""

import pandas as pd

# 模拟销售数据
data = {
    '区域': ['华东', '华东', '华北', '华北', '华东', '华北', '华南', '华南'],
    '产品': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B'],
    '月份': ['1月', '1月', '1月', '1月', '2月', '2月', '2月', '2月'],
    '销量': [100, 150, 80, 120, 120, 180, 90, 130],
    '销售额(万)': [50, 90, 40, 72, 60, 108, 45, 78]
}
df = pd.DataFrame(data)
print(df)

# 1. 基础透视表：按区域和产品分组，计算平均销量
pivot1 = pd.pivot_table(
    df,
    index='区域',       # 行：区域
    columns='产品',     # 列：产品
    values='销量',      # 值：销量
    aggfunc='mean'     # 聚合方式：平均值
)
print("1. 区域×产品的平均销量：")
print(pivot1, "\n")

# 2. 多指标透视表：同时计算销量总和与销售额均值
pivot2 = pd.pivot_table(
    df,
    index=['区域', '月份'],  # 多行索引：区域+月份
    columns='产品',
    values=['销量', '销售额(万)'],  # 多值：销量和销售额
    aggfunc={'销量': 'sum', '销售额(万)': 'mean'}  # 不同指标用不同聚合函数
)
print("2. 区域×月份×产品的多指标汇总：")
print(pivot2.round(1), "\n")

# 3. 包含总计的透视表
pivot3 = pd.pivot_table(
    df,
    index='区域',
    columns='产品',
    values='销量',
    aggfunc='sum',
    margins=True,      # 显示总计
    margins_name='合计'  # 总计名称
)
print("3. 带总计的销量汇总：")
print(pivot3)
