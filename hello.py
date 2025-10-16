import matplotlib.pyplot as plt
import pandas as pd


fig, axes = plt.subplots(1, 2, figsize=(10, 4))
plt.suptitle("Figure Super Title (plt.suptitle)", fontsize=16, color='blue') # 设置 Figure 总标题

# 使用 plt.title()
plt.sca(axes[0]) # 激活第一个子图
plt.plot([1, 2], [1, 2])
plt.title("Subplot 1 Title (plt.title)")

# 使用 ax.set_title()
axes[1].plot([1, 2], [2, 1])
axes[1].set_title("Subplot 2 Title (ax.set_title)", color='red')

plt.tight_layout(rect=(0, 0.03, 1, 0.95)) # 调整布局，防止suptitle与子图标题重叠
plt.show()
