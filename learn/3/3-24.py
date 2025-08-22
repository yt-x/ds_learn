# -*- coding: utf-8 -*-
"""
Created on Sat Jul 26 22:33:32 2025

@author: lch99
"""

import pandas as pd#例3-24
import matplotlib.pyplot as plt
import seaborn as sns
# 设置
sns.set_style("darkgrid")
sns.set_style({"font.sans-serif": "SimHei"})
# 读取数据
df = pd.read_csv(r"data/tip.csv")
# 绘制图表
sns.kdeplot(data=df, x="总额", hue="类型",multiple='stack')
# 显示
plt.show()
