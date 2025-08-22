# 先定义 cars 变量（汽车数据列表）
cars = [
    {"name": "卡罗拉", "price": 15.8},
    {"name": "本田雅阁", "price": 23.1},
    {"name": "哈弗H6", "price": 12.5}
]
# 按价格排序（key指定排序依据为price字段）
sorted_cars = sorted(cars, key=lambda x: x["price"], reverse=True)
# 打印排序结果
for car in sorted_cars:
    print(f"{car['name']}：{car['price']}万")