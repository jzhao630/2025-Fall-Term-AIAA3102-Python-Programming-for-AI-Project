import evaluate
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats
from .config import Config

# 设置绘图样式
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    acc = accuracy_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(
        predictions=predictions, references=labels, average="weighted"
    )

    return {"accuracy": acc["accuracy"], "f1": f1["f1"]}


def plot_test_results(test_results, predictions=None, labels=None, class_names=None, 
                      model_name="Model", output_dir=None, seed=None):
    """
    绘制测试结果的综合可视化
    
    Args:
        test_results: 包含测试指标的字典 (accuracy, f1, loss等)
        predictions: 模型预测结果 (numpy array)
        labels: 真实标签 (numpy array)
        class_names: 类别名称列表
        model_name: 模型名称
        output_dir: 输出目录
        seed: 随机种子
    """
    if output_dir is None:
        output_dir = Config.OUTPUT_DIR
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 创建大画布
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # ========== 1. 主要指标条形图 ==========
    ax1 = fig.add_subplot(gs[0, 0])
    metrics_names = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
    metrics_values = [
        test_results.get('eval_accuracy', test_results.get('accuracy', 0)),
        test_results.get('eval_f1', test_results.get('f1', 0)),
        test_results.get('eval_precision', test_results.get('precision', 0)),
        test_results.get('eval_recall', test_results.get('recall', 0))
    ]
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    bars = ax1.bar(metrics_names, metrics_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Score', fontweight='bold')
    ax1.set_title('Test Performance Metrics', fontsize=14, fontweight='bold')
    ax1.set_ylim([0, 1.1])
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # ========== 2. 性能雷达图 ==========
    ax2 = fig.add_subplot(gs[0, 1], projection='polar')
    
    categories = ['Accuracy', 'F1', 'Precision', 'Recall']
    values = metrics_values + [metrics_values[0]]  # 闭合多边形
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    
    ax2.plot(angles, values, 'o-', linewidth=2, color='#3498db', label=model_name)
    ax2.fill(angles, values, alpha=0.25, color='#3498db')
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(categories, fontsize=10)
    ax2.set_ylim(0, 1)
    ax2.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax2.set_title('Performance Radar', fontsize=14, fontweight='bold', pad=20)
    
    # ========== 3. 损失和时间信息 ==========
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis('off')
    
    info_text = f"""
    {'='*40}
    TEST RESULTS SUMMARY
    {'='*40}
    
    Model: {model_name}
    {'Seed: ' + str(seed) if seed else ''}
    
    PERFORMANCE METRICS
    {'─'*40}
    Accuracy:     {test_results.get('eval_accuracy', test_results.get('accuracy', 0)):.4f}
    F1 Score:     {test_results.get('eval_f1', test_results.get('f1', 0)):.4f}
    Precision:    {test_results.get('eval_precision', test_results.get('precision', 0)):.4f}
    Recall:       {test_results.get('eval_recall', test_results.get('recall', 0)):.4f}
    
    LOSS & EFFICIENCY
    {'─'*40}
    Test Loss:    {test_results.get('eval_loss', test_results.get('loss', 0)):.4f}
    Train Time:   {test_results.get('training_time', 0):.2f}s
    Samples/sec:  {test_results.get('samples_per_second', 'N/A')}
    
    ERROR ANALYSIS
    {'─'*40}
    Error Rate:   {(1 - test_results.get('eval_accuracy', test_results.get('accuracy', 0))):.4f}
    """
    
    ax3.text(0.1, 0.5, info_text, transform=ax3.transAxes,
            fontsize=10, verticalalignment='center', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # ========== 4. 混淆矩阵 (如果有预测结果) ==========
    if predictions is not None and labels is not None:
        from sklearn.metrics import confusion_matrix
        
        ax4 = fig.add_subplot(gs[1, :2])
        
        cm = confusion_matrix(labels, predictions)
        
        if class_names is None:
            class_names = [f'Class {i}' for i in range(len(cm))]
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names,
                   ax=ax4, cbar_kws={'label': 'Count'})
        
        ax4.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        ax4.set_ylabel('True Label', fontweight='bold')
        ax4.set_xlabel('Predicted Label', fontweight='bold')
        
        # ========== 5. 每类别性能 ==========
        ax5 = fig.add_subplot(gs[1, 2])
        
        from sklearn.metrics import classification_report
        report = classification_report(labels, predictions, target_names=class_names, output_dict=True)
        
        class_f1_scores = [report[name]['f1-score'] for name in class_names]
        class_support = [report[name]['support'] for name in class_names]
        
        y_pos = np.arange(len(class_names))
        bars = ax5.barh(y_pos, class_f1_scores, alpha=0.8, color='skyblue', edgecolor='black')
        ax5.set_yticks(y_pos)
        ax5.set_yticklabels(class_names)
        ax5.set_xlabel('F1 Score', fontweight='bold')
        ax5.set_title('Per-Class F1 Scores', fontsize=12, fontweight='bold')
        ax5.set_xlim([0, 1.1])
        ax5.grid(axis='x', alpha=0.3, linestyle='--')
        
        for i, (bar, support) in enumerate(zip(bars, class_support)):
            width = bar.get_width()
            ax5.text(width + 0.02, bar.get_y() + bar.get_height()/2.,
                    f'{width:.3f}\n(n={int(support)})',
                    ha='left', va='center', fontsize=9)
        
        # ========== 6. 预测分布 ==========
        ax6 = fig.add_subplot(gs[2, 0])
        
        unique, counts = np.unique(predictions, return_counts=True)
        ax6.bar(unique, counts, alpha=0.8, color='lightcoral', edgecolor='black')
        ax6.set_xlabel('Predicted Class', fontweight='bold')
        ax6.set_ylabel('Count', fontweight='bold')
        ax6.set_title('Prediction Distribution', fontsize=12, fontweight='bold')
        ax6.set_xticks(unique)
        if class_names:
            ax6.set_xticklabels([class_names[i] for i in unique], rotation=45, ha='right')
        ax6.grid(axis='y', alpha=0.3)
        
        # ========== 7. 真实标签分布 ==========
        ax7 = fig.add_subplot(gs[2, 1])
        
        unique, counts = np.unique(labels, return_counts=True)
        ax7.bar(unique, counts, alpha=0.8, color='lightgreen', edgecolor='black')
        ax7.set_xlabel('True Class', fontweight='bold')
        ax7.set_ylabel('Count', fontweight='bold')
        ax7.set_title('True Label Distribution', fontsize=12, fontweight='bold')
        ax7.set_xticks(unique)
        if class_names:
            ax7.set_xticklabels([class_names[i] for i in unique], rotation=45, ha='right')
        ax7.grid(axis='y', alpha=0.3)
        
        # ========== 8. 正确vs错误预测饼图 ==========
        ax8 = fig.add_subplot(gs[2, 2])
        
        correct = np.sum(predictions == labels)
        incorrect = len(labels) - correct
        
        colors_pie = ['#90EE90', '#FFB6C6']
        explode = (0.1, 0)
        
        wedges, texts, autotexts = ax8.pie(
            [correct, incorrect],
            explode=explode,
            labels=[f'Correct\n({correct})', f'Incorrect\n({incorrect})'],
            colors=colors_pie,
            autopct='%1.1f%%',
            shadow=True,
            startangle=90
        )
        
        for text in texts:
            text.set_fontsize(10)
            text.set_fontweight('bold')
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(11)
        
        ax8.set_title('Prediction Accuracy', fontsize=12, fontweight='bold')
    
    else:
        # 如果没有预测结果，显示提示信息
        ax4 = fig.add_subplot(gs[1:, :])
        ax4.axis('off')
        ax4.text(0.5, 0.5, 'No prediction data available for detailed visualization',
                ha='center', va='center', fontsize=16, style='italic',
                transform=ax4.transAxes)
    
    # 总标题
    main_title = f"{model_name} - Test Results"
    if seed is not None:
        main_title += f" (Seed: {seed})"
    fig.suptitle(main_title, fontsize=18, fontweight='bold', y=0.98)
    
    # 保存
    filename = f"test_results_{model_name.lower().replace(' ', '_')}"
    if seed is not None:
        filename += f"_seed_{seed}"
    filename += ".png"
    
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"\n✅ Test results visualization saved to: {filepath}")
    
    plt.show()
    

def create_performance_summary(results_list, output_dir=None):
    """
    创建多个实验的性能对比总结
    
    Args:
        results_list: 包含多个实验结果的列表
        output_dir: 输出目录
    """
    if output_dir is None:
        output_dir = Config.OUTPUT_DIR
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 提取数据
    exp_names = [r.get('model_type', f"Exp {i}") for i, r in enumerate(results_list)]
    accuracies = [r.get('accuracy', 0) for r in results_list]
    f1_scores = [r.get('f1', 0) for r in results_list]
    losses = [r.get('loss', 0) for r in results_list]
    times = [r.get('training_time', 0) for r in results_list]
    
    # 1. 准确率对比
    axes[0, 0].bar(exp_names, accuracies, alpha=0.8, color='skyblue', edgecolor='black')
    axes[0, 0].set_ylabel('Accuracy', fontweight='bold')
    axes[0, 0].set_title('Accuracy Comparison', fontweight='bold')
    axes[0, 0].set_ylim([0, 1.1])
    for i, v in enumerate(accuracies):
        axes[0, 0].text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold')
    
    # 2. F1分数对比
    axes[0, 1].bar(exp_names, f1_scores, alpha=0.8, color='lightcoral', edgecolor='black')
    axes[0, 1].set_ylabel('F1 Score', fontweight='bold')
    axes[0, 1].set_title('F1 Score Comparison', fontweight='bold')
    axes[0, 1].set_ylim([0, 1.1])
    for i, v in enumerate(f1_scores):
        axes[0, 1].text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold')
    
    # 3. 损失对比
    axes[1, 0].bar(exp_names, losses, alpha=0.8, color='lightgreen', edgecolor='black')
    axes[1, 0].set_ylabel('Loss', fontweight='bold')
    axes[1, 0].set_title('Loss Comparison', fontweight='bold')
    for i, v in enumerate(losses):
        axes[1, 0].text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold')
    
    # 4. 训练时间对比
    axes[1, 1].bar(exp_names, times, alpha=0.8, color='wheat', edgecolor='black')
    axes[1, 1].set_ylabel('Training Time (s)', fontweight='bold')
    axes[1, 1].set_title('Training Time Comparison', fontweight='bold')
    for i, v in enumerate(times):
        axes[1, 1].text(i, v + 1, f'{v:.1f}s', ha='center', fontweight='bold')
    
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, "performance_summary.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"\n✅ Performance summary saved to: {filepath}")
    
    plt.show()


def plot_metrics_comparison(metrics_dict, title="Metrics Comparison", output_dir=None):
    """
    绘制多个实验的指标对比图
    """
    if output_dir is None:
        output_dir = Config.OUTPUT_DIR
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 提取数据
    labels = list(metrics_dict.keys())
    accuracies = [metrics_dict[label]['accuracy'] for label in labels]
    f1_scores = [metrics_dict[label]['f1'] for label in labels]
    
    # 准确率条形图
    x_pos = np.arange(len(labels))
    bars1 = axes[0].bar(x_pos, accuracies, alpha=0.8, color='skyblue')
    axes[0].set_xlabel('Experiment')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Accuracy Comparison')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(labels, rotation=45, ha='right')
    axes[0].set_ylim([0, 1])
    
    # 在条形上添加数值
    for bar in bars1:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom')
    
    # F1分数条形图
    bars2 = axes[1].bar(x_pos, f1_scores, alpha=0.8, color='lightcoral')
    axes[1].set_xlabel('Experiment')
    axes[1].set_ylabel('F1 Score')
    axes[1].set_title('F1 Score Comparison')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(labels, rotation=45, ha='right')
    axes[1].set_ylim([0, 1])
    
    for bar in bars2:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    filename = os.path.join(output_dir, f"{title.lower().replace(' ', '_')}.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Metrics comparison plot saved to {filename}")
    plt.close()


def plot_confusion_matrix(predictions, labels, class_names=None, output_dir=None):
    """
    绘制混淆矩阵
    """
    if output_dir is None:
        output_dir = Config.OUTPUT_DIR
    
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    cm = confusion_matrix(labels, predictions)
    
    plt.figure(figsize=(10, 8))
    
    if class_names is None:
        class_names = [f'Class {i}' for i in range(len(cm))]
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    filename = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {filename}")
    plt.close()


def calculate_confidence_intervals(metrics_dict, confidence=0.95):
    """
    计算多个指标的置信区间
    metrics_dict: 字典，键为指标名，值为该指标的多次运行结果列表
    """
    results = {}
    
    for metric_name, values in metrics_dict.items():
        if len(values) < 2:
            continue
            
        n = len(values)
        mean = np.mean(values)
        std = np.std(values, ddof=1)
        
        # 计算t分布的置信区间
        if n >= 2:
            t_value = stats.t.ppf((1 + confidence) / 2, n - 1)
            margin_of_error = t_value * (std / np.sqrt(n))
            ci_lower = mean - margin_of_error
            ci_upper = mean + margin_of_error
        else:
            ci_lower = ci_upper = mean
        
        results[metric_name] = {
            "mean": mean,
            "std": std,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "n": n
        }
    
    return results


def print_confidence_intervals(results):
    """格式化打印置信区间结果"""
    print("\n" + "=" * 60)
    print("CONFIDENCE INTERVALS (95%)")
    print("=" * 60)
    
    for metric_name, stats in results.items():
        print(f"\n{metric_name.upper()}:")
        print(f"  Mean ± Std: {stats['mean']:.4f} ± {stats['std']:.4f}")
        print(f"  95% CI: [{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}]")
        print(f"  N: {stats['n']} runs")