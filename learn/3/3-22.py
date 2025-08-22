# -*- coding: utf-8 -*-
"""
Created on Sat Jul 26 22:24:36 2025

@author: lch99
"""

#例3-22
import seaborn as sns
import pandas as pd
#设置背景风格
sns.set_theme(style="darkgrid")
sns.set_style({"font.sans-serif": "SimHei"})
penguins=pd.read_csv("data/penguin.csv")
#penguins = sns.load_dataset("penguins")
# multiple：数量列表，stack堆叠，dodge避开
sns.displot(penguins, x="嘴喙长度", hue="种类", multiple="stack")
sns.displot(penguins, x="嘴喙长度", hue="性别", multiple="dodge")
