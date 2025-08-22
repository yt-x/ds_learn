# -*- coding: utf-8 -*-
"""
Created on Sat Jul 26 23:13:47 2025

@author: lch99
"""
import pandas as pd#例3-28
import matplotlib.pyplot as plt
import seaborn as sns# 设置
sns.set_style("whitegrid")
sns.set_style({"font.sans-serif": "SimHei"})# 画布大小
plt.figure(figsize=(9, 6))# 读取数据
df = pd.read_csv(r"data/flight.csv")# 使用透视表，重构DataFrame
df = df.pivot_table(index="年份", columns="月份", values="人数")# 行列转置
df = df.T# 绘制图表
ax=sns.lineplot(data=df)# 调整图例位置（注意这里是1，不再是1.2）
plt.legend(bbox_to_anchor=(1, 1))
ax.set_xticklabels(df.index,rotation=90)
ax.set_title('历年各月份乘机人数')
ax.set_ylabel('人数')# 显示plt.show()

