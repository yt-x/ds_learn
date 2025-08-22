# -*- coding: utf-8 -*-
"""
Created on Sat Jul 26 19:39:49 2025

@author: lch99
"""

#例3-19
import seaborn as sns
import pandas as pd
#设置背景风格
sns.set_theme(style="darkgrid")
sns.set_style({"font.sans-serif": "SimHei"})
penguins=pd.read_csv("data/penguin.csv")
#penguins = sns.load_dataset("penguins")
print(penguins)
sns.displot(penguins,x="嘴喙长度")
