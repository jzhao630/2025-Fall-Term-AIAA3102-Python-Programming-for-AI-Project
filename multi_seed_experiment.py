import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd
from .train import run_training
from .config import Config
import os


def calculate_confidence_interval(data, confidence=0.95):
    """
    计算置信区间
    """
    n = len(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1)  # 样本标准差
    
    if n < 2:
        return mean, (mean, mean)
    
    # 使用t分布计算置信区间
    t_value = stats.t.ppf((1 + confidence) / 2, n - 1)
    margin_of_error = t_value * (std / np.sqrt(n))
    
    return mean, (mean - margin_of_error, mean + margin_of_error)


def run_multi_seed_experiment(seeds=None, num_runs=3):
    """
    运行多随机种子实验并计算置信区间
    """
    if seeds is None:
        seeds = [42, 456, 101112]  # 默认的3个随机种子
    
    if len(seeds) > num_runs:
        seeds = seeds[:num_runs]
    
    print(f"Running multi-seed experiment with {len(seeds)} seeds: {seeds}")
    print("=" * 60)
    
    results = []
    all_metrics = {
        "accuracies": [],
        "f1_scores": [],
        "losses": [],
        "training_times": []
    }
    
    for i, seed in enumerate(seeds):
        print(f"\n\nRun {i+1}/{len(seeds)} - Seed: {seed}")
        print("-" * 40)
        
        try:
            # 运行训练
            result = run_training(seed=seed, use_early_stopping=True, subset_size=2400)
            results.append(result)
            
            # 收集指标
            all_metrics["accuracies"].append(result["accuracy"])
            all_metrics["f1_scores"].append(result["f1"])
            all_metrics["losses"].append(result["loss"])
            all_metrics["training_times"].append(result["training_time"])
            
            print(f"Run {i+1} completed - Accuracy: {result['accuracy']:.4f}, F1: {result['f1']:.4f}")
            
        except Exception as e:
            print(f"Run {i+1} with seed {seed} failed: {e}")
    
    if len(results) == 0:
        print("No successful runs!")
        return
    
    # 计算统计信息
    print("\n" + "=" * 60)
    print("MULTI-SEED EXPERIMENT RESULTS")
    print("=" * 60)
    
    # 创建结果DataFrame
    df_results = pd.DataFrame(results)
    print("\nDetailed Results:")
    print(df_results.to_string(index=False))
    
    # 计算每个指标的置信区间
    print("\n" + "=" * 60)
    print("CONFIDENCE INTERVALS (95%)")
    print("=" * 60)
    
    for metric_name, values in all_metrics.items():
        if len(values) > 1:
            mean, ci = calculate_confidence_interval(values)
            print(f"\n{metric_name.upper()}:")
            print(f"  Mean: {mean:.4f}")
            print(f"  Std: {np.std(values):.4f}")
            print(f"  95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")
            print(f"  Range: [{min(values):.4f}, {max(values):.4f}]")
        else:
            print(f"\n{metric_name.upper()}:")
            print(f"  Single value: {values[0]:.4f}")
    
    # 绘制结果可视化
    plot_multi_seed_results(all_metrics, seeds[:len(results)])
    
    # 保存结果到CSV
    output_dir = Config.OUTPUT_DIR
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    results_file = os.path.join(output_dir, "multi_seed_results.csv")
    df_results.to_csv(results_file, index=False)
    print(f"\nDetailed results saved to: {results_file}")
    
    # 保存汇总统计
    summary_stats = {}
    for metric_name, values in all_metrics.items():
        if len(values) > 0:
            summary_stats[f"{metric_name}_mean"] = np.mean(values)
            summary_stats[f"{metric_name}_std"] = np.std(values)
            if len(values) > 1:
                _, ci = calculate_confidence_interval(values)
                summary_stats[f"{metric_name}_ci_lower"] = ci[0]
                summary_stats[f"{metric_name}_ci_upper"] = ci[1]
    
    summary_file = os.path.join(output_dir, "multi_seed_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("MULTI-SEED EXPERIMENT SUMMARY\n")
        f.write("=" * 40 + "\n")
        f.write(f"Number of runs: {len(results)}\n")
        f.write(f"Seeds used: {seeds[:len(results)]}\n\n")
        
        for metric_name, values in all_metrics.items():
            if len(values) > 0:
                f.write(f"{metric_name.upper()}:\n")
                f.write(f"  Mean ± Std: {np.mean(values):.4f} ± {np.std(values):.4f}\n")
                if len(values) > 1:
                    _, ci = calculate_confidence_interval(values)
                    f.write(f"  95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]\n")
                f.write("\n")
    
    print(f"Summary statistics saved to: {summary_file}")
    
    return df_results, summary_stats


def plot_multi_seed_results(all_metrics, seeds):
    """
    绘制多种子实验结果可视化
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. 准确率分布
    axes[0, 0].boxplot(all_metrics["accuracies"])
    axes[0, 0].scatter(range(1, len(all_metrics["accuracies"]) + 1), 
                      all_metrics["accuracies"], color='red', s=50, zorder=3)
    axes[0, 0].set_title('Accuracy Distribution Across Seeds')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_xticks(range(1, len(seeds) + 1))
    axes[0, 0].set_xticklabels([f'Seed {s}' for s in seeds])
    
    # 2. F1分数分布
    axes[0, 1].boxplot(all_metrics["f1_scores"])
    axes[0, 1].scatter(range(1, len(all_metrics["f1_scores"]) + 1), 
                      all_metrics["f1_scores"], color='red', s=50, zorder=3)
    axes[0, 1].set_title('F1 Score Distribution Across Seeds')
    axes[0, 1].set_ylabel('F1 Score')
    axes[0, 1].set_xticks(range(1, len(seeds) + 1))
    axes[0, 1].set_xticklabels([f'Seed {s}' for s in seeds])
    
    # 3. 训练时间
    axes[1, 0].bar(range(len(all_metrics["training_times"])), all_metrics["training_times"])
    axes[1, 0].set_title('Training Time per Seed')
    axes[1, 0].set_xlabel('Seed')
    axes[1, 0].set_ylabel('Time (seconds)')
    axes[1, 0].set_xticks(range(len(seeds)))
    axes[1, 0].set_xticklabels(seeds)
    
    # 4. 准确率和F1的散点图
    axes[1, 1].scatter(all_metrics["accuracies"], all_metrics["f1_scores"], s=100)
    for i, seed in enumerate(seeds):
        axes[1, 1].annotate(f'Seed {seed}', 
                           (all_metrics["accuracies"][i], all_metrics["f1_scores"][i]),
                           xytext=(5, 5), textcoords='offset points')
    axes[1, 1].set_title('Accuracy vs F1 Score')
    axes[1, 1].set_xlabel('Accuracy')
    axes[1, 1].set_ylabel('F1 Score')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图像
    output_dir = Config.OUTPUT_DIR
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    plot_file = os.path.join(output_dir, "multi_seed_results.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {plot_file}")
    plt.show()


def bootstrap_confidence_interval(data, n_bootstrap=1000, confidence=0.95):
    """
    使用bootstrap方法计算置信区间
    """
    n = len(data)
    bootstrap_means = []
    
    for _ in range(n_bootstrap):
        # 有放回抽样
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_means.append(np.mean(sample))
    
    # 计算百分位数置信区间
    lower = np.percentile(bootstrap_means, (1 - confidence) / 2 * 100)
    upper = np.percentile(bootstrap_means, (1 + confidence) / 2 * 100)
    
    return np.mean(data), (lower, upper)


if __name__ == "__main__":
    # 示例：运行多种子实验
    results = run_multi_seed_experiment(num_runs=3)
    
    # 如果需要使用bootstrap方法
    print("\n" + "=" * 60)
    print("BOOTSTRAP CONFIDENCE INTERVALS (1000 resamples)")
    print("=" * 60)
    
    if results:
        df_results, _ = results
        
        # 使用bootstrap计算准确率的置信区间
        acc_mean, acc_ci = bootstrap_confidence_interval(
            df_results["accuracy"].values, 
            n_bootstrap=1000
        )
        print(f"\nAccuracy (Bootstrap):")
        print(f"  Mean: {acc_mean:.4f}")
        print(f"  95% CI: [{acc_ci[0]:.4f}, {acc_ci[1]:.4f}]")
        
        # 使用bootstrap计算F1的置信区间
        f1_mean, f1_ci = bootstrap_confidence_interval(
            df_results["f1"].values, 
            n_bootstrap=1000
        )
        print(f"\nF1 Score (Bootstrap):")
        print(f"  Mean: {f1_mean:.4f}")
        print(f"  95% CI: [{f1_ci[0]:.4f}, {f1_ci[1]:.4f}]")