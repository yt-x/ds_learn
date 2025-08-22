# -*- coding: utf-8 -*-
"""
Created on Sat Jul 26 22:45:12 2025

@author: lch99
"""

import pandas as pd#例3-19-2
import matplotlib.pyplot as plt
import seaborn as sns
# 设置
sns.set_style("darkgrid")
sns.set_style({"font.sans-serif": "SimHei"})
# 读取数据
df = pd.read_csv(r"data/tip.csv")
# 绘制图表
sns.regplot(data=df, x="总额", y="小费", color="orangered")
# 显示
plt.show()
