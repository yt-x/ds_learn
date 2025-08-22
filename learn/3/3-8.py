# -*- coding: utf-8 -*-
"""
Created on Sat Jul 26 18:10:22 2025

@author: lch99
"""

import seaborn as sns#例3-8
# 读入seaborn自身带的数据
fmri = sns.load_dataset("fmri")
print(fmri)
sns.relplot(x="timepoint", y="signal", hue="region", style="event",
            dashes=False, markers=True, kind="line", data=fmri)
