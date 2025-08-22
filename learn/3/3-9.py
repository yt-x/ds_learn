# -*- coding: utf-8 -*-  # 声明文件编码为utf-8
import seaborn as sns#例3-9
tips = sns.load_dataset("tips")
print(tips)
sns.relplot(x='total_bill', y='tip', data=tips)#例3-10
#例3-11
sns.relplot(x='total_bill', y='tip', data=tips, hue='smoker')
#例3-12
set(tips.day) # {‘Fri’, ‘Sat’, ‘Sun’, ‘Thur’}
sns.relplot(x='total_bill', y='tip', data=tips, hue='day')

#例3-13
sns.relplot(x='total_bill', y='tip', data=tips, style='sex')

#例3-14
sns.relplot(x='total_bill', y='tip',  data=tips, hue='smoker', style='sex')

#例3-15
sns.relplot(x='total_bill', y='tip', data=tips, hue='smoker', style='sex', size='size')
#例3-16
sns.relplot(x="total_bill", y="tip", col="sex", data=tips)
#3-17
sns.relplot(x="total_bill", y="tip", row="smoker", data=tips)
#例3-18
sns.relplot(x="total_bill", y="tip", row="smoker", col="sex", data=tips)
