# -*- coding: utf-8 -*-
"""
Created on Sat Jul 26 14:59:08 2025

@author: lch99
"""

#例3-4
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns# 设置
sns.set_style("darkgrid")
sns.set_style({"font.sans-serif": "SimHei"})# 读取数据
df = pd.read_csv(r"data/flight.csv")# 使用透视表，重构DataFrame
df = df.pivot_table(index="年份", columns="月份", values="人数")
# 绘制图表
sns.lineplot(data=df["1月"])# 显示
plt.show()
