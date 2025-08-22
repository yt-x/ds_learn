# -*- coding: utf-8 -*-
"""
Created on Sun Aug  3 22:20:24 2025

@author: lch99
"""

import jieba
import re
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd

# ----------------------
# 1. 配置与工具函数
# ----------------------
# 加载停用词（过滤无意义词汇）
def load_stopwords(filepath="stopwords.txt"):
    with open(filepath, "r", encoding="utf-8") as f:
        return set([line.strip() for line in f.readlines()])

stopwords = load_stopwords()

# 文本预处理（保留标点，全模式可能需要标点辅助分析）
def preprocess_text(text):
    """轻量预处理：仅去除多余空格，保留标点和特殊符号"""
    text = re.sub(r"\s+", " ", text).strip()
    return text

# 读取文本（支持字符串或文件）
def get_text(input_data):
    if isinstance(input_data, str) and input_data.endswith(".txt"):
        with open(input_data, "r", encoding="utf-8") as f:
            return preprocess_text(f.read())
    return preprocess_text(input_data)

# ----------------------
# 2. 全模式分词核心功能
# ----------------------
def full_mode_cut(text):
    """全模式分词（尽可能多的切分）"""
    # cut_all=True开启全模式
    full_words = jieba.lcut(text, cut_all=True)
    # 过滤停用词和空字符串
    filtered_full = [word for word in full_words if word and word not in stopwords]
    return full_words, filtered_full

def accurate_mode_cut(text):
    """精确模式分词（对比用）"""
    acc_words = jieba.lcut(text, cut_all=False)
    filtered_acc = [word for word in acc_words if word and word not in stopwords]
    return acc_words, filtered_acc

# ----------------------
# 3. 分词结果对比分析
# ----------------------
def compare_modes(full_words, acc_words):
    """对比全模式与精确模式的分词差异"""
    # 统计词数差异
    print(f"\n【模式对比】")
    print(f"全模式分词总数：{len(full_words)}（含重复和停用词）")
    print(f"精确模式分词总数：{len(acc_words)}（含重复和停用词）")
    
    # 提取全模式特有词语（在精确模式中未出现的词）
    acc_set = set(acc_words)
    full_unique = [word for word in full_words if word not in acc_set]
    print(f"全模式特有词语（前10个）：{full_unique[:10]}")
    
    # 生成对比表格
    compare_df = pd.DataFrame({
        "模式": ["全模式", "精确模式"],
        "分词数量": [len(full_words), len(acc_words)],
        "独特词语数": [len(set(full_unique)), 0],
        "平均词长": [round(sum(len(w) for w in full_words)/len(full_words), 2) if full_words else 0,
                   round(sum(len(w) for w in acc_words)/len(acc_words), 2) if acc_words else 0]
    })
    print("\n【分词模式对比表】")
    print(compare_df)
    return full_unique

# ----------------------
# 4. 歧义分析（基于全模式）
# ----------------------
def detect_ambiguity(text, full_filtered):
    """从全模式结果中检测潜在歧义短语"""
    # 歧义特征：同一位置有多种切分方式（通过全模式特有词与上下文组合判断）
    # 简单规则：全模式中出现的、长度≥2且在精确模式中未出现的词，可能对应歧义
    ambiguity_candidates = [word for word in full_filtered if len(word)>=2]
    # 统计高频歧义候选词
    ambiguity_counts = Counter(ambiguity_candidates).most_common(5)
    print("\n【潜在歧义短语（全模式特有高频词）】")
    for word, count in ambiguity_counts:
        # 查找词语在原文中的位置，展示上下文
        idx = text.find(word)
        if idx != -1:
            context = text[max(0, idx-5):min(len(text), idx+len(word)+5)]
            print(f"短语：{word}（出现{count}次），上下文：...{context}...")
    return ambiguity_candidates

# ----------------------
# 5. 全模式词频统计与可视化
# ----------------------
def full_mode_word_freq(filtered_full, topk=15):
    """统计全模式分词的词频（含重复切分）"""
    freq = Counter(filtered_full).most_common(topk)
    print(f"\n【全模式高频词（前{topk}）】")
    for word, count in freq:
        print(f"{word}: {count}次")
    return freq

def plot_freq(freq, title):
    """可视化词频"""
    words, counts = zip(*freq) if freq else ([], [])
    plt.figure(figsize=(12, 6))
    plt.bar(words, counts, color='orange')
    plt.xticks(rotation=45, ha='right')
    plt.title(title)
    plt.tight_layout()
    plt.show()

# ----------------------
# 6. 全模式N-gram分析（二元词）
# ----------------------
def full_mode_bigram(full_filtered):
    """从全模式结果中提取二元词（词语组合）"""
    bigrams = [f"{full_filtered[i]}-{full_filtered[i+1]}" 
               for i in range(len(full_filtered)-1)]
    bigram_freq = Counter(bigrams).most_common(10)
    print("\n【全模式高频二元词（前10）】")
    for bg, count in bigram_freq:
        print(f"{bg}: {count}次")
    return bigram_freq

# ----------------------
# 主函数：串联所有功能
# ----------------------
def full_mode_analyzer(input_text):
    # 1. 读取并预处理文本
    text = get_text(input_text)
    print(f"【分析文本预览】：{text[:100]}...")
    
    # 2. 两种模式分词
    full_words, filtered_full = full_mode_cut(text)
    acc_words, filtered_acc = accurate_mode_cut(text)
    
    # 3. 模式对比
    full_unique = compare_modes(full_words, acc_words)
    
    # 4. 歧义分析
    detect_ambiguity(text, filtered_full)
    
    # 5. 全模式词频统计与可视化
    full_freq = full_mode_word_freq(filtered_full)
    plot_freq(full_freq, "全模式分词高频词分布")
    
    # 6. 二元词分析
    bigram_freq = full_mode_bigram(filtered_full)
    plot_freq(bigram_freq, "全模式高频二元词分布")
    
    return {
        "full_words": full_words,
        "acc_words": acc_words,
        "full_freq": full_freq,
        "bigrams": bigram_freq
    }

# ----------------------
# 运行示例
# ----------------------
if __name__ == "__main__":
    # 示例文本（选择有歧义或多义词较多的文本，如新闻、散文）
    sample_text = """
    乒乓球拍卖完了。这件事的背后，是他和她的故事。
    他说：“我看见她那年，正是春暖花开的时候。”
    她则记得，那天的风很大，吹乱了她的头发，也吹乱了他的心。
    """
    # 运行分析
    result = full_mode_analyzer(sample_text)