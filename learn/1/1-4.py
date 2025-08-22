# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 13:06:03 2025

@author: lch99
"""

def calculate_annual_output(reactor):
    """计算单座核反应堆的年度发电量（单位：兆瓦时）"""
    # 发电量 = 装机容量(MW) × 年运行小时数 × 利用率
    capacity, hours, efficiency = reactor
    return round(capacity * hours * efficiency)

# 核反应堆数据：(装机容量MW, 年运行小时数, 利用率)
reactors = [
    (1200, 7000, 0.9),  # 反应堆A
    (1000, 7200, 0.88), # 反应堆B
    (1400, 6800, 0.92)  # 反应堆C
]

# 使用map计算所有反应堆的年度发电量
annual_outputs = list(map(calculate_annual_output, reactors))

# 输出结果
for i, output in enumerate(annual_outputs, 1):
    print(f"反应堆{i} 年度发电量: {output} 兆瓦时")
