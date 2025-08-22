# -*- coding: utf-8 -*-
"""
Created on Sun Aug  3 19:29:09 2025

@author: lch99
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体（解决中文显示问题）
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示异常的问题

fp = r"player-2023春季赛.csv"
df = pd.read_csv(fp)

data = df.sort_values("比赛场次", ascending=False)
data = data.iloc[:6]
print(data)
filter_cols = ["选手", "经济占比", "伤害占比", "承伤占比", "推塔占比", "参团率"]
data = data.loc[:, filter_cols]
print(data)
for col in filter_cols[1:]:
    data[col] = data[col].str.replace("%", "", regex=False)
    data[col] = data[col].astype("float")
data
N = 5 # 雷达图属性个数
angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
#将第一个角度值添加到角度数组的末尾，使得角度数组形成一个闭合的环形，用于雷达图的绘制。
angles = np.concatenate((angles, [angles[0]]))
fig = plt.figure(figsize=[10, 6])
for i in range(len(data)):
    values = data.iloc[i, 1:].tolist()
    #将第一个数据值再次添加到列表末尾，使数据形成一个闭合的环形，与角度数组对应。
    values.append(values[0])
    #生成一个字符串，表示当前子图在图形中的位置。例如，第一次循环时，position为231，第二次循环时为232，依此类推
    position = "23" + str(i + 1)
    #在图形中添加一个子图，使用生成的位置参数和polar=True表示创建一个极坐标子图，用于绘制雷达图。
    ax = fig.add_subplot(int(position), polar=True)
    #ax.plot绘制雷达图的线条，其中angles是角度数组，values是数据值数组，"o-"表示使用圆形标记和实线连接。
    ax.plot(angles, values, "o-")
    #填充雷达图的内部区域，设置透明度为0.4
    ax.fill(angles, values, alpha=0.4)
    #设置角度网格线的标签，将角度数组转换为度数，并使用数据集中的列名作为标签。
    ax.set_thetagrids(angles[:-1] * 180 / np.pi,
                      data.columns[1:].tolist())
    ax.set_title(data.iloc[i, 0], color="b")
    ax.set_ylim(0, 100)
#调整子图之间的垂直间距，这里设置为 0.5。
plt.subplots_adjust(hspace=0.5)
