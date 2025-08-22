# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 22:29:09 2025

@author: lch99
"""

# 列表去重
nums = [1, 2, 2, 3, 3, 3, 4]
unique_nums = list(set(nums))  # 先转集合去重，再转回列表
print(unique_nums)  # [1, 2, 3, 4]（注意：顺序可能变化）

# 字符串列表去重
words = ["apple", "banana", "apple", "cherry", "banana"]
unique_words = list(set(words))
print(unique_words)  # ["apple", "banana", "cherry"]（顺序不定）

# 保持去重后的顺序（Python 3.7+ 可用字典特性）
nums = [1, 2, 2, 3, 3, 3, 4]
unique_nums_ordered = list(dict.fromkeys(nums))  # 字典键唯一且保留插入顺序
print(unique_nums_ordered)  # [1, 2, 3, 4]（顺序与原列表一致）