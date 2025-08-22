# -*- coding: utf-8 -*-
"""
Created on Sat Jul 26 23:27:38 2025

@author: lch99
"""
import jieba
import jieba.analyse
import jieba.posseg as pseg  # 词性标注模块
from wordcloud import WordCloud
import matplotlib.pyplot as plt
# 待分词文本（核工业相关新闻片段）
text = """
核反应堆是核电站的核心设备，其安全运行直接关系到公众健康。
铀浓缩技术是核燃料生产的关键，需严格控制浓度以避免核扩散。
数字化监测系统提升了核设施的运行可靠性，保障核电安全。
"""

# ----------------------
# 1. 三种基础分词模式对比
# ----------------------
print("=== 1. 基础分词模式 ===")

# 精确模式（默认）：最常用，切分最准确，适合文本分析
exact_cut = jieba.lcut(text)
print("精确模式：", exact_cut)

# 全模式：把所有可能的词语都切分出来，可能有冗余
full_cut = jieba.lcut(text, cut_all=True)
print("全模式：   ", full_cut)

# 搜索引擎模式：在精确模式基础上，对长词进一步拆分（适合搜索引擎索引）
search_cut = jieba.lcut_for_search(text)
print("搜索引擎模式：", search_cut)


# ----------------------
# 2. 自定义词典（解决专业术语分词问题）
# ----------------------
print("\n=== 2. 自定义词典分词 ===")

# 问题：默认分词可能拆分专业术语（如“核反应堆”可能被拆分为“核/反应堆”）
# 解决：创建自定义词典（格式：词语 词频(可选) 词性(可选)）
with open("nuclear_dict.txt", "w", encoding="utf-8") as f:
    f.write("核反应堆 100 n\n")  # n表示名词
    f.write("铀浓缩 80 n\n")
    f.write("核燃料 60 n\n")

# 加载自定义词典
jieba.load_userdict("nuclear_dict.txt")
custom_cut = jieba.lcut(text)
print("加载专业词典后：", custom_cut)


# ----------------------
# 3. 词性标注（标记词语类型）
# ----------------------
print("\n=== 3. 词性标注 ===")
# 切分并标记词性（n：名词，v：动词，a：形容词等）
words_with_pos = pseg.lcut(text)
# 提取前5个词展示
print("词语 + 词性：", [(word, flag) for word, flag in words_with_pos[:5]])


# ----------------------
# 4. 去除停用词（过滤无意义词汇）
# ----------------------
print("\n=== 4. 去除停用词 ===")

# 定义停用词表（无实际意义的词）
stopwords = {"的", "是", "其", "和", "以", "直接", "到", "需"}

# 先分词，再过滤停用词
cut_words = jieba.lcut(text)
filtered_words = [word for word in cut_words if word not in stopwords]
print("分词后（含停用词）：", cut_words[:10])  # 前10个
print("过滤后（去停用词）：", filtered_words[:10])  # 前10个


# ----------------------
# 5. 关键词提取（基于TF-IDF）
# ----------------------
print("\n=== 5. 关键词提取 ===")

# 从文本中提取前3个关键词（带权重）
keywords = jieba.analyse.extract_tags(
    text,
    topK=3,        # 提取数量
    withWeight=True,  # 返回权重
    allowPOS=()    # 允许的词性（默认所有）
)

print("关键词（权重越高越重要）：")
for word, weight in keywords:
    print(f"{word}：{weight:.4f}")
    
#例4-2 词云图
print("\n=== 6. 词云图 ===")
# 1. 将jieba分词并过滤停用词后的词拼接成词云所需的空格分隔格式
word_list = " ".join(filtered_words)  
# 2. 生成词云（设置参数）
wc = WordCloud(
    font_path="simhei.ttf",  # 中文字体（需确保本地有该字体，或替换为其他中文字体路径）
    background_color="white",  # 背景色
    max_words=20,  # 最多显示20个词
    width=800, height=400  # 尺寸
)
wordcloud = wc.generate(word_list)  # 根据词语生成词云
# 3. 显示词云
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud)
plt.axis("off")  # 隐藏坐标轴
plt.show()