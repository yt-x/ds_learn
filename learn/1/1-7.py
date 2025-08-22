# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 22:39:45 2025

@author: lch99
"""

# 模拟商品A数据
view_users_A = {"u1", "u2", "u3", "u4", "u5", "u6"}
cart_users_A = {"u2", "u3", "u5", "u6", "u7"}
buy_users_A = {"u2", "u5", "u7"}
# 1. 浏览但未加入购物车的用户（潜在流失用户）
view_not_cart_A = view_users_A - cart_users_A
print("浏览未加购：", view_not_cart_A)  # {'u1', 'u4'}
# 2. 加入购物车但未购买的用户（高潜力转化用户）
cart_not_buy_A = cart_users_A - buy_users_A
print("加购未购买：", cart_not_buy_A)  # {'u3', 'u6'}
# 3. 完整转化路径用户（浏览→加购→购买）
full_path_A = view_users_A & cart_users_A & buy_users_A
print("完整转化用户：", full_path_A)  # {'u2', 'u5'}
# 4. 购买用户占浏览用户的比例（转化率）
conversion_rate_A = len(buy_users_A & view_users_A) / len(view_users_A) * 100
print(f"浏览到购买转化率：{conversion_rate_A:.1f}%")  # 33.3%
# 模拟商品B数据
view_users_B = {"u1", "u2", "u4", "u6", "u7"}
cart_users_B = { "u2", "u6", "u7"}
buy_users_B = {"u2", "u6"}
# 5. 潜在关联用户
# 计算共同浏览用户（交集）
common_users = view_users_A & view_users_B
print(f"同时浏览过A和B的用户: {common_users}，共 {len(common_users)} 人")
# 计算A到B的相关性
if len(view_users_A) > 0:
    corr_A_to_B = len(common_users) / len(view_users_A)
    print(f"商品A到B的相关性: {corr_A_to_B:.2f}（{corr_A_to_B*100:.1f}%）")
else:
    print("商品A没有浏览用户，无法计算相关性")
# 计算B到A的相关性
if len(view_users_B) > 0:
    corr_B_to_A = len(common_users) / len(view_users_B)
    print(f"商品B到A的相关性: {corr_B_to_A:.2f}（{corr_B_to_A*100:.1f}%）")
else:
    print("商品B没有浏览用户，无法计算相关性")
# 对比分析
print("\n相关性对比分析:")
if corr_A_to_B > corr_B_to_A:
    print(f"A到B的相关性（{corr_A_to_B*100:.1f}%）高于B到A（{corr_B_to_A*100:.1f}%）")
elif corr_A_to_B < corr_B_to_A:
    print(f"B到A的相关性（{corr_B_to_A*100:.1f}%）高于A到B（{corr_A_to_B*100:.1f}%）")
else:
    print(f"A到B与B到A的相关性相同（均为{corr_A_to_B*100:.1f}%）")