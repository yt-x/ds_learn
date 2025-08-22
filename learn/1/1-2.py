# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 12:28:50 2025

@author: lch99
"""

# 公共充电桩费用计算（简洁版）
calc_charge = lambda capacity, base, service, period: capacity * (
    base + service + (0.3 if period == 'peak' else -0.2 if period == 'valley' else 0)
)
# 参数定义
battery_capacity = 50  # 电池容量(kWh)
base_price = 0.8       # 基础电价(元/度)
service_fee = 0.7      # 服务费(元/度)
# 计算不同时段费用
periods = {'peak': '高峰', 'flat': '平峰', 'valley': '低谷'}
for period, name in periods.items():
    cost = calc_charge(battery_capacity, base_price, service_fee, period)
    print(f"{name}时段充电费用: {cost:.2f}元")
    