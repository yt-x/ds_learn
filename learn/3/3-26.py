# -*- coding: utf-8 -*-
"""
Created on Sat Jul 26 22:38:53 2025

@author: lch99
"""

import pandas as pd#例3-26
import matplotlib.pyplot as plt
import seaborn as sns
# 设置
sns.set_style("darkgrid")
sns.set_style({"font.sans-serif": "SimHei"})
# 读取数据
df = pd.read_csv(r"data/tip.csv")
# 绘制图表
sns.stripplot(data=df, x="时间", y="总额", color="black")
sns.boxenplot(data=df, x="时间", y="总额")
# 显示
plt.show()
