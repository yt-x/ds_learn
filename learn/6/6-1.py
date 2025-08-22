# -*- coding: utf-8 -*-
"""
Created on Sun Aug  3 18:41:19 2025

@author: lch99
"""

from random import randrange

# 模拟历史商品评分数据，一共20个用户，商品一共15个，每个用户对4到9个商品进行随机评分
# 每个商品的评分最低1分最高5分
data = {'user'+str(i):{'product'+str(randrange(1, 16)):randrange(1, 6)
                       for j in range(randrange(4, 10))}
        for i in range(20)}

# 模拟当前用户评分数据，为5种随机商品评分
user = {'product'+str(randrange(1, 16)):randrange(1,6) for i in range(5)}
# 最相似的用户及其对商品评分情况，这个 lambda 函数返回一个元组，作为判断用户相似度的 “排序键”
# 第一个值：两个用户共同评分的商品的数量的负数
# 第二个值：sum(((历史用户评分 - 当前用户评分)²) for 共同商品)
f = lambda item:(-len(item[1].keys()&user.keys()),
                 sum(((item[1].get(product)-user.get(product))**2
                      for product in user.keys()&item[1].keys())))
similarUser, products = min(data.items(), key=f)

# 在输出结果中，第一列表示两个人共同评分的商品的数量
# 第二列表示用户与当前用户的评分平方误差和（值越小越相似）
# 然后是该用户对商品的评分数据
print('known data'.center(50, '='))
for item in data.items():
    print(len(item[1].keys()&user.keys()),
          sum(((item[1].get(product)-user.get(product))**2
               for product in user.keys()&item[1].keys())),
          item,
          sep=':')
print('current user'.center(50, '='))
print(user)
print('most similar user and his products'.center(50, '='))
print(similarUser, products, sep=':')
print('recommended product'.center(50, '='))
# 在当前用户没购买过的商品中选择评分最高的进行推荐
print(max(products.keys()-user.keys(), key=lambda product: products[product]))
