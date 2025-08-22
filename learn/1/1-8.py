import pandas as pd
# 1. 创建汽车销售的Series（不同维度的数据）
# 车型指导价（万元）
guide_price = pd.Series(
    data=[12.98, 17.98, 25.58, 37.98, 13.98],
    index=["飞度", "雅阁", "冠道", "奥德赛", "凌派"]
)
# 2. 查看Series基本信息
print("=== 广汽本田车型指导价（万元） ===")
print(guide_price)
print("\n=== 数据类型 ===", guide_price.dtype)
print("=== 索引 ===", guide_price.index.tolist())

# 3. 基本操作：取值与筛选
print("\n=== 雅阁的指导价 ===", guide_price["雅阁"], "万元")
print("=== 价格在20万元以上的车型 ===")
print(guide_price[guide_price > 20])

# 4. 运算：计算指导价的10%购置税（近似）
tax = guide_price * 0.1
tax.name = "购置税（万元，近似）"  # 给Series命名
print("\n=== 各车型购置税（万元） ===")
print(tax.round(2))  # 保留两位小数

# 5. 组合查询：获取冠道的完整信息
print("\n=== 冠道完整信息 ===")
print(f"车型：冠道")
print(f"指导价：{guide_price['冠道']}万元")
# 6. 切片查询：
print("\n=== 切片信息 ===")
print(f"1到3行")
print(f"{guide_price.iloc[1:4]}")
print(f"1，3行")
print(f"{guide_price.iloc[[1,3]]}")
print(f"前3行")
print(f"{guide_price.iloc[:3]}")
print(f"后3行")
print(f"{guide_price.iloc[-3:]}")
print(f"输出飞度和凌派")
print(f"{guide_price.loc[['飞度','凌派']]}")
# 7. 统计分析
print("\n=== 价格统计 ===")
print(f"平均指导价：{guide_price.mean():.2f}万元")
print(f"最高指导价：{guide_price.max()}万元（车型：{guide_price.idxmax()}）")
print(f"最低指导价：{guide_price.min()}万元（车型：{guide_price.idxmin()}）")
# 8. 数据修改与删除
print("\n=== 数据修改 ===")
print("\n雅阁涨价10%")
guide_price['雅阁']=guide_price['雅阁']*1.1
print(guide_price)
print("\n删除雅阁")
guide_price.pop('雅阁')
print(guide_price)

