# 先定义 cars 变量（汽车数据列表）
cars = [
    {"name": "比亚迪 唐", "price": 18.8},
    {"name": "本田雅阁", "price": 23.1},
    {"name": "哈弗H6", "price": 12.5}
]

lowprice_cars = filter(lambda car :15<=car["price"]<=20, cars)
print([car["name"] for car in lowprice_cars]) 