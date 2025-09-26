import torch
import matplotlib.pyplot as plt
import numpy as np

# --- 配置区 ---
# 确保这个文件名与您在 main.py 中保存的文件名一致
ACCURACY_FILE = 'final_accuracy_cifar10_EB.pt'
OUTPUT_IMAGE_FILE = 'my_accuracy_plot.png'

# --- 绘图脚本 ---
try:
    # 1. 加载保存的准确率数据
    accuracy_tensor = torch.load(ACCURACY_FILE)

    # 2. 将PyTorch Tensor转换为NumPy数组，以便绘图
    # .cpu()确保数据在CPU上，.numpy()进行转换
    accuracy_data = accuracy_tensor.cpu().numpy()

    # 3. 创建x轴的数据（代表epochs轮数）
    # arange的起始点是1，结束点是数据长度+1，这样x轴就从1开始了
    epochs = np.arange(1, len(accuracy_data) + 1)

    # 4. 开始绘图
    print(f"成功加载 {len(accuracy_data)} 轮的准确率数据。正在生成图表...")

    # 创建一个图形窗口，可以指定尺寸
    plt.figure(figsize=(8, 6))

    # 绘制准确率曲线
    plt.plot(epochs, accuracy_data, label='AdaAggRL (Your Result)', color='red')

    # 5. 美化图表，使其看起来像示例图
    plt.title('Model Accuracy vs. Epochs (EB Attack)')  # 添加标题
    plt.xlabel('epochs')                                 # 设置x轴标签
    plt.ylabel('accuracy')                               # 设置y轴标签
    plt.ylim(0, 1.0)                                     # 设置y轴范围从0到1
    plt.grid(True, linestyle='--', alpha=0.5)             # 添加淡淡的网格线
    plt.legend()                                         # 显示图例

    # 6. 保存图表到文件并显示出来
    plt.savefig(OUTPUT_IMAGE_FILE)
    print(f"图表已成功保存为: {OUTPUT_IMAGE_FILE}")

    plt.show()

except FileNotFoundError:
    print(f"错误：找不到文件 '{ACCURACY_FILE}'。请确保文件名正确，并且文件与此脚本在同一个目录下。")
except Exception as e:
    print(f"发生了一个错误: {e}")