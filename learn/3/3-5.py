# -*- coding: utf-8 -*-
"""
Created on Sat Jul 26 15:06:12 2025

@author: lch99
"""

import pandas as pd#例3-5
import matplotlib.pyplot as plt
import seaborn as sns# 设置
sns.set_style("darkgrid")
sns.set_style({"font.sans-serif": "SimHei"})# 读取数据
df = pd.read_csv(r"data/flight.csv")# 使用透视表，重构DataFrame
df = df.pivot_table(index="年份", columns="月份", values="人数")# 调整顺序
orders = [str(i)+"月" for i in range(1, 13)]
df = df[orders]
df.to_excel(r"data/2.xlsx")# 行列转置
df = df.T
# 绘制图表  
sns.lineplot(data=df)
# 显示
plt.show()
