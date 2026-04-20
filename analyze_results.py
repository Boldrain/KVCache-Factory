import pandas as pd
import matplotlib.pyplot as plt
import os

# ==========================
# 1 读取数据
# ==========================

file_path = "all_results.csv"

df = pd.read_csv(file_path)

# 指标列
metrics = [
    "time_sec",
    "mem_avg_MB",
    "mem_peak_MB",
    "power_avg_W",
    "power_peak_W",
    "time_per_output_token",
    "time_to_first_token",
]

# ==========================
# 2 计算平均值
# ==========================

grouped = df.groupby(["kv", "quant"])[metrics].mean()

print("\n===== 每种 (kv, quant) 组合平均值 =====\n")
print(grouped)

# 保存平均值表
grouped.to_csv("kv_quant_average_metrics.csv")

# ==========================
# 3 画柱状图
# ==========================

os.makedirs("figures", exist_ok=True)

# 将 index 变为列
grouped_reset = grouped.reset_index()

# 创建组合标签
grouped_reset["config"] = grouped_reset["kv"] + "-" + grouped_reset["quant"]

for metric in metrics:

    plt.figure(figsize=(10,6))

    plt.bar(grouped_reset["config"], grouped_reset[metric])

    plt.title(f"{metric} comparison")
    plt.xlabel("KV + Quantization")
    plt.ylabel(metric)

    plt.xticks(rotation=45)

    plt.tight_layout()

    save_path = f"figures/{metric}_bar.png"
    plt.savefig(save_path)

    plt.close()

print("\n柱状图已保存到 figures/ 文件夹")