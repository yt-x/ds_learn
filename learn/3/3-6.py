# -*- coding: utf-8 -*-
"""
Created on Sat Jul 26 15:17:27 2025

@author: lch99
"""

import numpy as np#例3-6
import pandas as pd
import seaborn as sns
# 二维数组 cumsum累计和
df = pd.DataFrame(dict(time=np.arange(500), value=np.random.randn(500).cumsum()))
g = sns.relplot(x="time", y="value", kind="line", data=df)
# 更改x显示方式，斜着显示
g.fig.autofmt_xdate()
