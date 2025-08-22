# -*- coding: utf-8 -*-
"""
Created on Sat Jul 26 22:47:50 2025

@author: lch99
"""

import pandas as pd#例2-19-3
import matplotlib.pyplot as plt
import seaborn as sns
# 设置
sns.set_style("darkgrid")
sns.set_style({"font.sans-serif": "SimHei"})
# 读取数据
df = pd.read_csv(r"data/penguin.csv")
# 绘制图表
sns.pairplot(data=df, hue="性别", markers=["o", "s"])
# 显示
plt.show()
