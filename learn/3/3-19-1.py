# -*- coding: utf-8 -*-
"""
Created on Sat Jul 26 22:17:24 2025

@author: lch99
"""

import pandas as pd#例3-19-1
import matplotlib.pyplot as plt
import seaborn as sns# 设置
sns.set_style("darkgrid")
sns.set_style({"font.sans-serif": "SimHei"})# 读取数据
df = pd.read_csv(r"data/flight.csv")# 重构DataFrame（透视表）
df = df.pivot_table(index="年份", columns="月份", values="人数")
# 行列转置
df = df.T
# 绘制图表
sns.heatmap(data=df, annot=True, fmt=".2f")
# 显示
plt.show()
