import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

# 设置字体为 Times New Roman
rcParams['font.family'] = 'Times New Roman'

# 定义混淆矩阵的数据（可以手动修改）
data = np.array([
    [1322, 136, 4, 3, 1],
    [0, 3265, 50, 13, 0],
    [10, 276, 3171, 443, 1],
    [0, 1, 190, 2059, 5],
    [0, 0, 77, 967, 514]
])

# 计算总体精度 A, 精度 P, 召回率 R, 和 F1 分数
def calculate_metrics(confusion_matrix):
    # 提取主要统计量
    true_positives = np.diag(confusion_matrix)  # 对角线上的值
    false_positives = np.sum(confusion_matrix, axis=0) - true_positives
    false_negatives = np.sum(confusion_matrix, axis=1) - true_positives
    total_samples = np.sum(confusion_matrix)
    overall_accuracy = np.sum(true_positives) / total_samples

    # 总体精度
    overall_accuracy = np.sum(true_positives) / total_samples

    # 每一类的 Precision, Recall, 和 F1 Score
    precision = true_positives / (true_positives + false_positives + 1e-8)
    recall = true_positives / (true_positives + false_negatives + 1e-8)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)

    # 计算均值
    precision_mean = np.mean(precision)
    recall_mean = np.mean(recall)
    f1_mean = np.mean(f1_score)

    # 输出结果
    print(f"Overall Accuracy (A): {overall_accuracy:.4f}")
    print(f"Mean Precision (P): {precision_mean:.4f}")
    print(f"Mean Recall (R): {recall_mean:.4f}")
    print(f"Mean F1 Score: {f1_mean:.4f}")

    return overall_accuracy, precision, recall, f1_score

# 调用计算函数
overall_accuracy, precision, recall, f1_score = calculate_metrics(data)

# 创建图表
plt.figure(figsize=(8, 8))  # 设置正方形图像
ax = sns.heatmap(data, annot=True, fmt="d", cmap="YlGnBu", cbar=False, 
                 annot_kws={"size": 20, "color": "black"}, linewidths=0.5, linecolor='white', square=True)

# 设置标签
ax.set_xlabel("Predicted labels", fontsize=20, labelpad=10)
ax.set_ylabel("True labels", fontsize=20, labelpad=10)

# 设置刻度和字体大小
ax.set_xticks(np.arange(data.shape[1]) + 0.5)
ax.set_yticks(np.arange(data.shape[0]) + 0.5)
ax.set_xticklabels([1, 2, 3, 4, 5], fontsize=20)
ax.set_yticklabels([1, 2, 3, 4, 5], fontsize=20)

# 调整网格线
ax.tick_params(left=False, bottom=False)

# 显示图表
plt.tight_layout()
plt.show()