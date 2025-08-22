import pandas as pd


# 数据读取语预处理
df = pd.read_csv(r'learn\1.2 作业1要求及资源数据\orders.csv')

df = df.drop_duplicates(subset=['order_id'], keep='first')


df['order_date'] = df['order_date'].astype('datetime64[ns]')
df['month'] = pd.to_datetime(df['order_date'], format="%Y-%m-%d")

# print(df.info())
# print(df)

# 缺失值与格式处理

df['user_id'] = df['user_id'].fillna("未知_" + df['order_id'].astype(str))

df['amount'] = df['amount'].str.replace('¥', '').astype(float)
df['amount'] =df['amount'].fillna(df['amount'].median())


# 异常值处理
df = df.drop(df[df['amount'] <= 0].index)
df['is_high'] = df['amount'] > 10000
# print(df['is_high'])



# 统计分析
print(df.groupby('payment_method')[['order_id', 'amount']].agg({'order_id':'count', "amount":"sum"}))

print(df.groupby('status')['amount'].mean())

print(df.groupby(df['month'].dt.month)['order_id'].count())


# print(df)