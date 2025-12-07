import numpy as np
import random
from datasets import Dataset
import matplotlib.pyplot as plt
from scipy import stats
from .config import Config
from .train import run_short_training
import os
import pandas as pd
import torch
import logging

# Set up logging
logger = logging.getLogger(__name__)


class FewShotLearner:
    """
    Few-shot学习器
    实现少样本学习和零样本学习
    """
    
    def __init__(self):
        self.num_classes = Config.NUM_LABELS
    
    def create_few_shot_dataset(self, full_dataset: Dataset, n_shots: int, seed: int = None) -> Dataset:
        """
        创建few-shot数据集
        从每个类别中抽取n_shots个样本
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # 获取标签 - 将张量转换为整数
        label_key = 'labels' if 'labels' in full_dataset.column_names else 'label'
        labels = full_dataset[label_key]
        
        # 如果标签是张量，转换为Python整数列表
        if isinstance(labels[0], torch.Tensor):
            labels = [label.item() for label in labels]
        elif hasattr(labels[0], 'numpy'):  # 如果是numpy数组
            labels = [int(label) for label in labels]
        
        # 按类别分组
        class_indices = {i: [] for i in range(self.num_classes)}
        for idx, label in enumerate(labels):
            class_indices[label].append(idx)
        
        # 从每个类别中抽取样本
        selected_indices = []
        for class_id, indices in class_indices.items():
            if len(indices) >= n_shots:
                selected = random.sample(indices, n_shots)
            else:
                selected = indices  # 如果样本不足，使用所有样本
            selected_indices.extend(selected)
        
        # 打乱顺序
        random.shuffle(selected_indices)
        
        # 创建新的数据集（保持所有字段）
        few_shot_dataset = full_dataset.select(selected_indices)
        
        logger.info(f"Created few-shot dataset: {len(few_shot_dataset)} samples "
                   f"({n_shots} per class, {self.num_classes} classes)")
        
        return few_shot_dataset
    
    def create_balanced_dataset(self, full_dataset: Dataset, n_samples: int, seed: int = None) -> Dataset:
        """
        创建平衡的数据集（每个类别样本数大致相等）
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # 获取标签 - 将张量转换为整数
        label_key = 'labels' if 'labels' in full_dataset.column_names else 'label'
        labels = full_dataset[label_key]
        
        # 如果标签是张量，转换为Python整数列表
        if isinstance(labels[0], torch.Tensor):
            labels = [label.item() for label in labels]
        elif hasattr(labels[0], 'numpy'):  # 如果是numpy数组
            labels = [int(label) for label in labels]
        
        # 按类别分组
        class_indices = {i: [] for i in range(self.num_classes)}
        for idx, label in enumerate(labels):
            class_indices[label].append(idx)
        
        # 计算每个类别应该抽取的样本数
        samples_per_class = n_samples // self.num_classes
        remainder = n_samples % self.num_classes
        
        selected_indices = []
        for i, (class_id, indices) in enumerate(class_indices.items()):
            n_to_select = samples_per_class + (1 if i < remainder else 0)
            if len(indices) >= n_to_select:
                selected = random.sample(indices, n_to_select)
            else:
                selected = indices  # 如果样本不足，使用所有样本
            selected_indices.extend(selected)
        
        # 打乱顺序
        random.shuffle(selected_indices)
        
        # 创建新的数据集（保持所有字段）
        balanced_dataset = full_dataset.select(selected_indices)
        
        logger.info(f"Created balanced dataset: {len(balanced_dataset)} samples "
                   f"({self.num_classes} classes, target {n_samples} samples)")
        
        return balanced_dataset


def run_zero_shot_experiment(test_dataset, tokenizer):
    """
    运行零样本学习实验
    """
    print("\n" + "=" * 60)
    print("ZERO-SHOT LEARNING EXPERIMENT")
    print("=" * 60)
    
    # 加载预训练模型（不进行微调）
    from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
    from .utils import compute_metrics
    
    print("Loading pre-trained model for zero-shot evaluation...")
    model = AutoModelForSequenceClassification.from_pretrained(
        Config.MODEL_NAME, num_labels=Config.NUM_LABELS
    )
    
    # 检查测试数据集大小
    if len(test_dataset) == 0:
        raise ValueError("Test dataset is empty")    
    
    # 配置评估参数
    eval_args = TrainingArguments(
        output_dir=os.path.join(Config.OUTPUT_DIR, "zero_shot"),
        per_device_eval_batch_size=Config.BATCH_SIZE,
        seed=Config.SEED,
        report_to="none",
        disable_tqdm=False,
    )
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # 使用小部分测试数据
    test_size = min(500, len(test_dataset))
    small_test = test_dataset.select(range(test_size))
    
    trainer = Trainer(
        model=model,
        args=eval_args,
        eval_dataset=small_test,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # 评估
    results = trainer.evaluate()
    
    print(f"\nZero-shot Results:")
    print(f"  Accuracy: {results['eval_accuracy']:.4f}")
    print(f"  F1 Score: {results['eval_f1']:.4f}")
    print(f"  Loss: {results['eval_loss']:.4f}")

    result_dict = {
        'accuracy': results['eval_accuracy'],
        'f1': results['eval_f1'],
        'loss': results['eval_loss'],
        'method': 'zero_shot'
    }

    # 绘制零样本结果
    plot_zero_shot_results(result_dict)

    return result_dict


def run_few_shot_experiment(train_dataset, val_dataset, test_dataset, tokenizer, n_shots_list=None):
    """
    运行少样本学习实验
    """
    if n_shots_list is None:
        n_shots_list = [1, 5, 10, 50, 100]
    
    print("\n" + "=" * 60)
    print("FEW-SHOT LEARNING EXPERIMENT")
    print("=" * 60)
    
    learner = FewShotLearner()
    results = []
    
    for n_shots in n_shots_list:
        print(f"\nTraining with {n_shots}-shot learning")
        print("-" * 40)
        
        try:
            # 创建few-shot数据集
            few_shot_train = learner.create_few_shot_dataset(
                train_dataset, 
                n_shots=n_shots,
                seed=Config.SEED
            )
            
            print(f"Created dataset with {len(few_shot_train)} samples "
                  f"({n_shots} per class, {Config.NUM_LABELS} classes)")
            
            # 运行训练
            result = run_short_training(
                train_dataset=few_shot_train,
                val_dataset=val_dataset,
                test_dataset=test_dataset,
                tokenizer=tokenizer,
                num_epochs=10  # Few-shot通常需要更多epoch
            )
            
            result['n_shots'] = n_shots
            result['method'] = f'few_shot_{n_shots}'
            results.append(result)
            
            print(f"Results - Accuracy: {result['accuracy']:.4f}, F1: {result['f1']:.4f}")
            
        except Exception as e:
            print(f"Failed to train with {n_shots}-shot: {e}")
            continue
    
    # 分析结果
    if results:
        plot_few_shot_results(results)
    
    return results


def run_scaling_analysis(train_dataset, val_dataset, test_dataset, tokenizer, sample_sizes=None):
    """
    运行数据缩放分析（学习曲线）
    """
    if sample_sizes is None:
        sample_sizes = [200, 500, 1000, 2000, 5000, 20000]
    
    print("\n" + "=" * 60)
    print("SCALING ANALYSIS (LEARNING CURVES)")
    print("=" * 60)
    
    learner = FewShotLearner()
    results = []
    
    for size in sample_sizes:
        print(f"\nTraining with {size} samples")
        print("-" * 40)
        
        try:
            # 创建平衡的子集
            sampled_train = learner.create_balanced_dataset(
                train_dataset,
                n_samples=size,
                seed=Config.SEED
            )
            
            print(f"Training on {len(sampled_train)} samples")
            
            # 根据数据量调整训练轮数
            if size >= 5000:
                num_epochs = 3
            elif size >= 1000:
                num_epochs = 5
            else:
                num_epochs = 10  # 小数据集需要更多轮数
            
            # 运行训练
            result = run_short_training(
                train_dataset=sampled_train,
                val_dataset=val_dataset,
                test_dataset=test_dataset,
                tokenizer=tokenizer,
                num_epochs=num_epochs
            )
            
            result['sample_size'] = size
            result['method'] = f'scaling_{size}'
            results.append(result)
            
            print(f"Results - Accuracy: {result['accuracy']:.4f}, F1: {result['f1']:.4f}")
            
        except Exception as e:
            print(f"Failed to train with {size} samples: {e}")
            continue
    
    # 分析结果
    if results:
        plot_scaling_results(results)
        fit_learning_curve(results)
    
    return results

def plot_zero_shot_results(zero_shot_result):
    """
    绘制零样本学习结果的可视化
    
    Args:
        zero_shot_result: 包含零样本实验结果的字典
    """
    if not zero_shot_result:
        print("No zero-shot results to plot")
        return
    
    print("\n" + "=" * 60)
    print("ZERO-SHOT RESULTS VISUALIZATION")
    print("=" * 60)
    
    # 提取指标
    accuracy = zero_shot_result.get('accuracy', 0.0)
    f1 = zero_shot_result.get('f1', 0.0)
    loss = zero_shot_result.get('loss', 0.0)
    
    # 创建图表
    fig = plt.figure(figsize=(15, 5))
    
    # 1. 条形图 - 准确率和F1分数
    ax1 = plt.subplot(1, 3, 1)
    metrics = ['Accuracy', 'F1 Score']
    values = [accuracy, f1]
    colors = ['skyblue', 'lightcoral']
    
    bars = ax1.bar(metrics, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Score')
    ax1.set_title('Zero-Shot Performance Metrics', fontsize=12, fontweight='bold')
    ax1.set_ylim([0, 1])
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 在条形图上添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.4f}',
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # 2. 饼图 - 性能分布
    ax2 = plt.subplot(1, 3, 2)
    
    # 计算正确和错误的比例
    correct_ratio = accuracy
    incorrect_ratio = 1 - accuracy
    
    labels = [f'Correct\n({correct_ratio*100:.1f}%)', 
              f'Incorrect\n({incorrect_ratio*100:.1f}%)']
    sizes = [correct_ratio, incorrect_ratio]
    colors_pie = ['#90EE90', '#FFB6C6']
    explode = (0.1, 0)  # 突出显示正确的部分
    
    wedges, texts, autotexts = ax2.pie(sizes, explode=explode, labels=labels, 
                                        colors=colors_pie, autopct='%1.1f%%',
                                        shadow=True, startangle=90)
    
    # 美化文本
    for text in texts:
        text.set_fontsize(10)
        text.set_fontweight('bold')
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(10)
    
    ax2.set_title('Prediction Distribution', fontsize=12, fontweight='bold')
    
    # 3. 指标雷达图（与理想性能对比）
    ax3 = plt.subplot(1, 3, 3, projection='polar')
    
    # 定义指标
    categories = ['Accuracy', 'F1 Score', 'Robustness\n(1-Loss)']
    
    # 零样本性能
    zero_shot_values = [
        accuracy,
        f1,
        max(0, 1 - min(loss, 1))  # 将loss转换为正向指标
    ]
    
    # 理想性能（用于对比）
    ideal_values = [1.0, 1.0, 1.0]
    
    # 计算角度
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    zero_shot_values += zero_shot_values[:1]  # 闭合多边形
    ideal_values += ideal_values[:1]
    angles += angles[:1]
    
    # 绘制
    ax3.plot(angles, ideal_values, 'o-', linewidth=2, label='Ideal', color='green', alpha=0.3)
    ax3.fill(angles, ideal_values, alpha=0.1, color='green')
    
    ax3.plot(angles, zero_shot_values, 'o-', linewidth=2, label='Zero-Shot', color='blue')
    ax3.fill(angles, zero_shot_values, alpha=0.25, color='blue')
    
    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels(categories, fontsize=9)
    ax3.set_ylim(0, 1)
    ax3.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax3.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax3.set_title('Performance Radar', fontsize=12, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    # 保存图像
    output_dir = Config.OUTPUT_DIR
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    plot_file = os.path.join(output_dir, "zero_shot_results.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"\n✅ Zero-shot results visualization saved to: {plot_file}")
    
    # 打印详细统计
    print("\n" + "-" * 60)
    print("ZERO-SHOT DETAILED STATISTICS")
    print("-" * 60)
    print(f"  Accuracy:        {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  F1 Score:        {f1:.4f} ({f1*100:.2f}%)")
    print(f"  Loss:            {loss:.4f}")
    print(f"  Error Rate:      {(1-accuracy):.4f} ({(1-accuracy)*100:.2f}%)")
    
    # 性能评级
    if accuracy >= 0.9:
        rating = "Excellent ⭐⭐⭐⭐⭐"
    elif accuracy >= 0.8:
        rating = "Very Good ⭐⭐⭐⭐"
    elif accuracy >= 0.7:
        rating = "Good ⭐⭐⭐"
    elif accuracy >= 0.6:
        rating = "Fair ⭐⭐"
    else:
        rating = "Needs Improvement ⭐"
    
    print(f"  Performance:     {rating}")
    print("-" * 60)
    
    plt.show()
    

def plot_few_shot_results(results):
    """
    绘制few-shot学习结果的综合可视化
    """
    if not results:
        print("No results to plot")
        return
    
    print("\n" + "=" * 60)
    print("FEW-SHOT RESULTS VISUALIZATION")
    print("=" * 60)
    
    # 提取数据
    n_shots = [r['n_shots'] for r in results]
    accuracies = [r['accuracy'] for r in results]
    f1_scores = [r['f1'] for r in results]
    losses = [r.get('loss', 0) for r in results]
    
    # 创建大画布
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # ========== 1. 准确率曲线 (对数尺度) ==========
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(n_shots, accuracies, 'o-', linewidth=2, markersize=10, color='#3498db')
    ax1.fill_between(n_shots, accuracies, alpha=0.3, color='#3498db')
    ax1.set_xlabel('Number of Shots per Class', fontweight='bold')
    ax1.set_ylabel('Accuracy', fontweight='bold')
    ax1.set_title('Accuracy vs Number of Shots', fontsize=12, fontweight='bold')
    ax1.set_xscale('log')
    ax1.set_ylim([0, 1.1])
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # 在图上标注值
    for x, y in zip(n_shots, accuracies):
        ax1.annotate(f'{y:.3f}', (x, y), textcoords="offset points", 
                    xytext=(0, 10), ha='center', fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))
    
    # ========== 2. F1分数曲线 (对数尺度) ==========
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(n_shots, f1_scores, 's-', linewidth=2, markersize=10, color='#e74c3c')
    ax2.fill_between(n_shots, f1_scores, alpha=0.3, color='#e74c3c')
    ax2.set_xlabel('Number of Shots per Class', fontweight='bold')
    ax2.set_ylabel('F1 Score', fontweight='bold')
    ax2.set_title('F1 Score vs Number of Shots', fontsize=12, fontweight='bold')
    ax2.set_xscale('log')
    ax2.set_ylim([0, 1.1])
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    for x, y in zip(n_shots, f1_scores):
        ax2.annotate(f'{y:.3f}', (x, y), textcoords="offset points", 
                    xytext=(0, 10), ha='center', fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))
    
    # ========== 3. 性能雷达图 (比较不同shot数) ==========
    ax3 = fig.add_subplot(gs[0, 2], projection='polar')
    
    categories = ['Accuracy', 'F1', 'Robustness']
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    
    # 选择几个代表性的shot数进行比较
    representative_shots = [min(n_shots), n_shots[len(n_shots)//2], max(n_shots)]
    colors_radar = ['#3498db', '#2ecc71', '#e74c3c']
    
    for shot, color in zip(representative_shots, colors_radar):
        idx = n_shots.index(shot)
        values = [
            accuracies[idx],
            f1_scores[idx],
            max(0, 1 - min(losses[idx], 1))
        ]
        values += values[:1]
        
        ax3.plot(angles, values, 'o-', linewidth=2, label=f'{shot}-shot', color=color)
        ax3.fill(angles, values, alpha=0.15, color=color)
    
    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels(categories, fontsize=10)
    ax3.set_ylim(0, 1)
    ax3.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax3.set_title('Performance Comparison', fontsize=12, fontweight='bold', pad=20)
    
    # ========== 4. 性能提升热力图 ==========
    ax4 = fig.add_subplot(gs[1, 0])
    
    # 计算相对于最小shot数的提升
    baseline_acc = accuracies[0]
    baseline_f1 = f1_scores[0]
    
    improvements = []
    for i in range(len(n_shots)):
        acc_improvement = (accuracies[i] - baseline_acc) / baseline_acc * 100 if baseline_acc > 0 else 0
        f1_improvement = (f1_scores[i] - baseline_f1) / baseline_f1 * 100 if baseline_f1 > 0 else 0
        improvements.append([acc_improvement, f1_improvement])
    
    improvements = np.array(improvements).T
    
    import seaborn as sns
    sns.heatmap(improvements, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
               xticklabels=[f'{s}-shot' for s in n_shots],
               yticklabels=['Accuracy', 'F1 Score'],
               ax=ax4, cbar_kws={'label': 'Improvement (%)'})
    ax4.set_title('Performance Improvement vs Baseline', fontsize=12, fontweight='bold')
    
    # ========== 5. 条形图对比 ==========
    ax5 = fig.add_subplot(gs[1, 1])
    
    x_pos = np.arange(len(n_shots))
    width = 0.35
    
    bars1 = ax5.bar(x_pos - width/2, accuracies, width, label='Accuracy', 
                    alpha=0.8, color='skyblue', edgecolor='black')
    bars2 = ax5.bar(x_pos + width/2, f1_scores, width, label='F1 Score',
                    alpha=0.8, color='lightcoral', edgecolor='black')
    
    ax5.set_xlabel('Number of Shots', fontweight='bold')
    ax5.set_ylabel('Score', fontweight='bold')
    ax5.set_title('Accuracy & F1 Comparison', fontsize=12, fontweight='bold')
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels([f'{s}' for s in n_shots])
    ax5.set_ylim([0, 1.1])
    ax5.legend()
    ax5.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    # ========== 6. 详细统计信息 ==========
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    # 找到最佳表现
    best_idx = accuracies.index(max(accuracies))
    worst_idx = accuracies.index(min(accuracies))
    
    # 计算平均提升
    avg_improvement = (max(accuracies) - min(accuracies)) / (len(n_shots) - 1) if len(n_shots) > 1 else 0
    
    info_text = f"""
    {'='*40}
    FEW-SHOT LEARNING SUMMARY
    {'='*40}
    
    OVERALL STATISTICS
    {'─'*40}
    Total Experiments:    {len(n_shots)}
    Shot Range:           {min(n_shots)} - {max(n_shots)}
    
    BEST PERFORMANCE
    {'─'*40}
    Best Configuration:   {n_shots[best_idx]}-shot
    Accuracy:             {accuracies[best_idx]:.4f}
    F1 Score:             {f1_scores[best_idx]:.4f}
    
    WORST PERFORMANCE
    {'─'*40}
    Configuration:        {n_shots[worst_idx]}-shot
    Accuracy:             {accuracies[worst_idx]:.4f}
    F1 Score:             {f1_scores[worst_idx]:.4f}
    
    IMPROVEMENT ANALYSIS
    {'─'*40}
    Total Improvement:    {(max(accuracies) - min(accuracies))*100:.2f}%
    Avg per Step:         {avg_improvement*100:.2f}%
    
    RECOMMENDATION
    {'─'*40}
    Optimal Shot Number:  {n_shots[best_idx]}
    Expected Accuracy:    {accuracies[best_idx]:.4f}
    """
    
    ax6.text(0.05, 0.5, info_text, transform=ax6.transAxes,
            fontsize=9, verticalalignment='center', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # 总标题
    fig.suptitle('Few-Shot Learning Comprehensive Analysis', 
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    # 保存图像
    output_dir = Config.OUTPUT_DIR
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    plot_file = os.path.join(output_dir, "few_shot_results.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"\n✅ Few-shot results visualization saved to: {plot_file}")
    
    # 保存数据
    df = pd.DataFrame(results)
    data_file = os.path.join(output_dir, "few_shot_data.csv")
    df.to_csv(data_file, index=False)
    print(f"✅ Few-shot data saved to: {data_file}")
    
    plt.show()


def plot_scaling_results(results):
    """
    绘制数据缩放分析结果的综合可视化
    """
    if not results:
        print("No results to plot")
        return
    
    print("\n" + "=" * 60)
    print("SCALING ANALYSIS VISUALIZATION")
    print("=" * 60)
    
    # 提取数据并排序
    sorted_results = sorted(results, key=lambda x: x['sample_size'])
    sample_sizes = [r['sample_size'] for r in sorted_results]
    accuracies = [r['accuracy'] for r in sorted_results]
    f1_scores = [r['f1'] for r in sorted_results]
    losses = [r['loss'] for r in sorted_results]
    training_times = [r.get('training_time', 0) for r in sorted_results]
    
    # 创建大画布
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # ========== 1. 学习曲线 - 准确率 (对数尺度) ==========
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(sample_sizes, accuracies, 'o-', linewidth=2.5, markersize=10, 
            color='#3498db', label='Accuracy')
    ax1.fill_between(sample_sizes, accuracies, alpha=0.3, color='#3498db')
    ax1.set_xlabel('Training Set Size', fontweight='bold', fontsize=11)
    ax1.set_ylabel('Accuracy', fontweight='bold', fontsize=11)
    ax1.set_title('Learning Curve: Accuracy', fontsize=13, fontweight='bold')
    ax1.set_xscale('log')
    ax1.set_ylim([0, 1.1])
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend()
    
    # 标注关键点
    for i, (x, y) in enumerate(zip(sample_sizes, accuracies)):
        if i == 0 or i == len(sample_sizes) - 1:  # 标注首尾
            ax1.annotate(f'{y:.3f}', (x, y), textcoords="offset points",
                        xytext=(0, 10), ha='center', fontsize=9, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))
    
    # ========== 2. 学习曲线 - F1分数 (对数尺度) ==========
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(sample_sizes, f1_scores, 's-', linewidth=2.5, markersize=10,
            color='#e74c3c', label='F1 Score')
    ax2.fill_between(sample_sizes, f1_scores, alpha=0.3, color='#e74c3c')
    ax2.set_xlabel('Training Set Size', fontweight='bold', fontsize=11)
    ax2.set_ylabel('F1 Score', fontweight='bold', fontsize=11)
    ax2.set_title('Learning Curve: F1 Score', fontsize=13, fontweight='bold')
    ax2.set_xscale('log')
    ax2.set_ylim([0, 1.1])
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend()
    
    for i, (x, y) in enumerate(zip(sample_sizes, f1_scores)):
        if i == 0 or i == len(sample_sizes) - 1:
            ax2.annotate(f'{y:.3f}', (x, y), textcoords="offset points",
                        xytext=(0, 10), ha='center', fontsize=9, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))
    
    # ========== 3. 损失曲线 (对数尺度) ==========
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(sample_sizes, losses, '^-', linewidth=2.5, markersize=10,
            color='#9b59b6', label='Loss')
    ax3.fill_between(sample_sizes, losses, alpha=0.3, color='#9b59b6')
    ax3.set_xlabel('Training Set Size', fontweight='bold', fontsize=11)
    ax3.set_ylabel('Loss', fontweight='bold', fontsize=11)
    ax3.set_title('Learning Curve: Loss', fontsize=13, fontweight='bold')
    ax3.set_xscale('log')
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.legend()
    
    # ========== 4. 训练时间 vs 数据量 ==========
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(sample_sizes, training_times, 'd-', linewidth=2.5, markersize=10,
            color='#2ecc71', label='Training Time')
    ax4.fill_between(sample_sizes, training_times, alpha=0.3, color='#2ecc71')
    ax4.set_xlabel('Training Set Size', fontweight='bold', fontsize=11)
    ax4.set_ylabel('Training Time (seconds)', fontweight='bold', fontsize=11)
    ax4.set_title('Training Time vs Data Size', fontsize=13, fontweight='bold')
    ax4.set_xscale('log')
    ax4.grid(True, alpha=0.3, linestyle='--')
    ax4.legend()
    
    # ========== 5. 性能雷达图 (比较不同数据量) ==========
    ax5 = fig.add_subplot(gs[1, 1], projection='polar')
    
    categories = ['Accuracy', 'F1', 'Robustness', 'Efficiency']
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    
    # 选择代表性的数据量
    indices = [0, len(sample_sizes)//2, -1]  # 最小、中间、最大
    colors_radar = ['#3498db', '#f39c12', '#e74c3c']
    labels_radar = [f'{sample_sizes[i]} samples' for i in indices]
    
    max_time = max(training_times) if max(training_times) > 0 else 1
    
    for idx, color, label in zip(indices, colors_radar, labels_radar):
        values = [
            accuracies[idx],
            f1_scores[idx],
            max(0, 1 - min(losses[idx], 1)),
            1 - (training_times[idx] / max_time)  # 效率指标
        ]
        values += values[:1]
        
        ax5.plot(angles, values, 'o-', linewidth=2, label=label, color=color)
        ax5.fill(angles, values, alpha=0.15, color=color)
    
    ax5.set_xticks(angles[:-1])
    ax5.set_xticklabels(categories, fontsize=10)
    ax5.set_ylim(0, 1)
    ax5.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax5.grid(True, alpha=0.3)
    ax5.legend(loc='upper right', bbox_to_anchor=(1.4, 1.1), fontsize=9)
    ax5.set_title('Multi-Metric Comparison', fontsize=13, fontweight='bold', pad=20)
    
    # ========== 6. 边际收益分析 ==========
    ax6 = fig.add_subplot(gs[1, 2])
    
    # 计算边际准确率提升
    marginal_gains = []
    for i in range(1, len(accuracies)):
        gain = (accuracies[i] - accuracies[i-1]) / (sample_sizes[i] - sample_sizes[i-1])
        marginal_gains.append(gain * 1000)  # 每1000个样本的提升
    
    x_marginal = sample_sizes[1:]
    ax6.bar(range(len(marginal_gains)), marginal_gains, alpha=0.8, 
           color='coral', edgecolor='black')
    ax6.set_xlabel('Data Size Increment', fontweight='bold', fontsize=11)
    ax6.set_ylabel('Marginal Accuracy Gain\n(per 1000 samples)', fontweight='bold', fontsize=10)
    ax6.set_title('Marginal Returns Analysis', fontsize=13, fontweight='bold')
    ax6.set_xticks(range(len(marginal_gains)))
    ax6.set_xticklabels([f'{sample_sizes[i]}\n→\n{sample_sizes[i+1]}' 
                         for i in range(len(marginal_gains))], fontsize=7, rotation=0)
    ax6.grid(axis='y', alpha=0.3)
    ax6.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    
    # ========== 7. 效率分析 (样本/秒 vs 数据量) ==========
    ax7 = fig.add_subplot(gs[2, 0])
    
    samples_per_second = [size / time if time > 0 else 0 
                         for size, time in zip(sample_sizes, training_times)]
    
    ax7.plot(sample_sizes, samples_per_second, 'o-', linewidth=2.5, 
            markersize=10, color='#16a085')
    ax7.set_xlabel('Training Set Size', fontweight='bold', fontsize=11)
    ax7.set_ylabel('Throughput (samples/sec)', fontweight='bold', fontsize=11)
    ax7.set_title('Training Efficiency', fontsize=13, fontweight='bold')
    ax7.set_xscale('log')
    ax7.grid(True, alpha=0.3, linestyle='--')
    
    # ========== 8. 热力图 - 综合性能 ==========
    ax8 = fig.add_subplot(gs[2, 1])
    
    # 归一化所有指标到 [0, 1]
    norm_acc = np.array(accuracies)
    norm_f1 = np.array(f1_scores)
    norm_loss = 1 - np.array(losses) / max(losses) if max(losses) > 0 else np.ones(len(losses))
    norm_time = 1 - np.array(training_times) / max(training_times) if max(training_times) > 0 else np.ones(len(training_times))
    
    heatmap_data = np.array([norm_acc, norm_f1, norm_loss, norm_time])
    
    import seaborn as sns
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlGnBu',
               xticklabels=[f'{s}' for s in sample_sizes],
               yticklabels=['Accuracy', 'F1', 'Robustness', 'Efficiency'],
               ax=ax8, cbar_kws={'label': 'Normalized Score'})
    ax8.set_title('Comprehensive Performance Heatmap', fontsize=13, fontweight='bold')
    ax8.set_xlabel('Training Set Size', fontweight='bold', fontsize=11)
    
    # ========== 9. 详细统计信息 ==========
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    
    # 找到最佳配置
    best_idx = accuracies.index(max(accuracies))
    
    # 计算总体提升
    total_improvement = (max(accuracies) - min(accuracies)) * 100
    
    # 计算收益递减点
    diminishing_idx = None
    if len(marginal_gains) >= 2:
        threshold = np.mean(marginal_gains) * 0.3
        for i, gain in enumerate(marginal_gains):
            if gain < threshold and i > 0:
                diminishing_idx = i + 1
                break
    
    info_text = f"""
    {'='*42}
    SCALING ANALYSIS SUMMARY
    {'='*42}
    
    DATA RANGE
    {'─'*42}
    Min Samples:          {min(sample_sizes):,}
    Max Samples:          {max(sample_sizes):,}
    Total Experiments:    {len(sample_sizes)}
    
    BEST PERFORMANCE
    {'─'*42}
    Sample Size:          {sample_sizes[best_idx]:,}
    Accuracy:             {accuracies[best_idx]:.4f}
    F1 Score:             {f1_scores[best_idx]:.4f}
    Training Time:        {training_times[best_idx]:.1f}s
    
    IMPROVEMENT METRICS
    {'─'*42}
    Total Improvement:    {total_improvement:.2f}%
    Improvement Rate:     {total_improvement/(len(sample_sizes)-1):.2f}% per step
    
    EFFICIENCY ANALYSIS
    {'─'*42}
    Best Throughput:      {max(samples_per_second):.1f} samples/s
    Avg Throughput:       {np.mean(samples_per_second):.1f} samples/s
    """
    
    if diminishing_idx:
        info_text += f"""
    DIMINISHING RETURNS
    {'─'*42}
    Starts at:            ~{sample_sizes[diminishing_idx]:,} samples
    """
    
    info_text += f"""
    RECOMMENDATION
    {'─'*42}
    Optimal Size:         {sample_sizes[best_idx]:,} samples
    Expected Accuracy:    {accuracies[best_idx]:.4f}
    Cost-Benefit Ratio:   {'Excellent' if best_idx < len(sample_sizes)//2 else 'Good'}
    """
    
    ax9.text(0.05, 0.5, info_text, transform=ax9.transAxes,
            fontsize=8.5, verticalalignment='center', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # 总标题
    fig.suptitle('Data Scaling Analysis - Comprehensive Report', 
                fontsize=18, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    # 保存图像
    output_dir = Config.OUTPUT_DIR
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    plot_file = os.path.join(output_dir, "scaling_analysis_results.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"\n✅ Scaling analysis visualization saved to: {plot_file}")
    
    # 保存数据
    df = pd.DataFrame(sorted_results)
    data_file = os.path.join(output_dir, "scaling_analysis_data.csv")
    df.to_csv(data_file, index=False)
    print(f"✅ Scaling analysis data saved to: {data_file}")
    
    plt.show()


def fit_learning_curve(results):
    """
    拟合学习曲线并分析
    """
    if len(results) < 3:
        print("Not enough data points for learning curve fitting")
        return
    
    print("\n" + "=" * 60)
    print("LEARNING CURVE ANALYSIS")
    print("=" * 60)
    
    # 提取数值型数据
    numeric_results = sorted(results, key=lambda x: x['sample_size'])
    
    sample_sizes = [r['sample_size'] for r in numeric_results]
    accuracies = [r['accuracy'] for r in numeric_results]

    # 添加数值稳定性检查, 避免准确率为0或1的情况（导致log无穷大）
    accuracies = np.clip(accuracies, 0.01, 0.99)
    
    # 转换为对数尺度
    log_sizes = np.log(sample_sizes)
    try:
        log_accuracies = np.log(accuracies / (1 - accuracies))  # logit转换
    except (RuntimeWarning, ValueError) as e:
        logger.warning(f"Logit transformation warning: {e}")
        # 使用线性转换作为备用
        log_accuracies = accuracies
    
    try:
        # 拟合线性模型
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_sizes, log_accuracies)
        
        print(f"\nLearning Curve Parameters:")
        print(f"  Slope: {slope:.4f}")
        print(f"  Intercept: {intercept:.4f}")
        print(f"  R-squared: {r_value**2:.4f}")
        print(f"  p-value: {p_value:.4f}")
        
        # 解释结果
        if slope > 0:
            print(f"  Interpretation: Positive learning curve slope ({slope:.4f})")
            print(f"    - Model benefits from more data")
            
            # 估计达到特定准确率所需的数据量
            target_accuracies = [0.8, 0.85, 0.9, 0.95]
            for target_acc in target_accuracies:
                if 0 < target_acc < 1:
                    logit_target = np.log(target_acc / (1 - target_acc))
                    estimated_size = np.exp((logit_target - intercept) / slope)
                    
                    # ✅ 添加: 检查估计值的合理性
                    if 0 < estimated_size < 1e9:  # 避免异常大的值
                        print(f"    - Estimated data needed for {target_acc*100:.0f}% accuracy: {int(estimated_size):,} samples")
        else:
            print(f"  Interpretation: Negative or flat learning curve slope ({slope:.4f})")
            print(f"    - Model may not benefit significantly from more data")
        
        # 计算收益递减点
        if len(accuracies) >= 3:
            # 计算边际收益
            marginal_gains = []
            for i in range(1, len(accuracies)):
                gain = accuracies[i] - accuracies[i-1]
                relative_gain = gain / (sample_sizes[i] - sample_sizes[i-1])
                marginal_gains.append(relative_gain)
            
            # 找到收益明显下降的点
            if len(marginal_gains) >= 2:
                threshold = np.mean(marginal_gains) * 0.3
                diminishing_point = None
                for i, gain in enumerate(marginal_gains):
                    if gain < threshold and i > 0:
                        diminishing_point = int(sample_sizes[i+1])
                        break
                
                if diminishing_point:
                    print(f"\nDiminishing Returns Analysis:")
                    print(f"  Diminishing returns start around: {diminishing_point:,} samples")
                    print(f"  Beyond this point, adding more data provides limited improvement")
        
        # 绘制拟合曲线
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 原始数据
        ax.plot(sample_sizes, accuracies, 'o-', label='Observed', linewidth=2, markersize=8)
        
        # 拟合曲线
        x_fit = np.linspace(min(sample_sizes), max(sample_sizes) * 1.2, 100)
        log_x_fit = np.log(x_fit)
        log_y_fit = slope * log_x_fit + intercept
        
        # ✅ 修复: 安全的逆logit变换
        try:
            y_fit = 1 / (1 + np.exp(-log_y_fit))
            y_fit = np.clip(y_fit, 0, 1)  # 确保在[0,1]范围内
        except (RuntimeWarning, ValueError):
            y_fit = np.clip(log_y_fit, 0, 1)  # 备用方案
        
        ax.plot(x_fit, y_fit, '--', label=f'Fitted (R²={r_value**2:.3f})', linewidth=2)
        
        ax.set_xlabel('Training Set Size (log scale)')
        ax.set_ylabel('Accuracy')
        ax.set_title('Learning Curve with Power Law Fit')
        ax.set_xscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图像
        output_dir = Config.OUTPUT_DIR
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        plot_file = os.path.join(output_dir, "learning_curve_fit.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"\nLearning curve fit saved to: {plot_file}")
        plt.show()
        
    except Exception as e:
        logger.error(f"Failed to fit learning curve: {e}")
        import traceback
        traceback.print_exc()


def compare_generalization_methods(zero_shot_result, few_shot_results, scaling_results):
    """
    比较不同的泛化方法
    """
    print("\n" + "=" * 60)
    print("GENERALIZATION METHODS COMPARISON")
    print("=" * 60)

    # ✅ 添加: 检查输入数据
    if not zero_shot_result:
        logger.warning("No zero-shot results available")
        zero_shot_result = {'accuracy': 0.0, 'f1': 0.0, 'method': 'zero_shot'}
    
    # 准备数据
    methods = ['Zero-shot']
    accuracies = [zero_shot_result['accuracy']]
    f1_scores = [zero_shot_result['f1']]
    
    # 添加few-shot的最佳结果
    if few_shot_results:
        best_few_shot = max(few_shot_results, key=lambda x: x['accuracy'])
        methods.append(f"Few-shot ({best_few_shot['n_shots']}-shot)")
        accuracies.append(best_few_shot['accuracy'])
        f1_scores.append(best_few_shot['f1'])
    
    # 添加完整微调结果
    if scaling_results:
        # 找到最大的样本量（接近完整数据集）
        max_size_result = max(scaling_results, key=lambda x: x['sample_size'])
        methods.append(f'Full Fine-tuning ({max_size_result["sample_size"]} samples)')
        accuracies.append(max_size_result['accuracy'])
        f1_scores.append(max_size_result['f1'])
    
    # 检查是否有足够的数据进行比较
    if len(methods) < 2:
        print("Not enough results to compare. Need at least 2 methods.")
        return

    # 绘制比较图
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    x_pos = np.arange(len(methods))
    
    # 准确率条形图
    bars1 = axes[0].bar(x_pos, accuracies, alpha=0.8, color='skyblue')
    axes[0].set_xlabel('Method')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Accuracy by Generalization Method')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(methods, rotation=45, ha='right')
    axes[0].set_ylim([0, 1])
    
    for bar in bars1:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom')
    
    # F1分数条形图
    bars2 = axes[1].bar(x_pos, f1_scores, alpha=0.8, color='lightcoral')
    axes[1].set_xlabel('Method')
    axes[1].set_ylabel('F1 Score')
    axes[1].set_title('F1 Score by Generalization Method')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(methods, rotation=45, ha='right')
    axes[1].set_ylim([0, 1])
    
    for bar in bars2:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # 保存图像
    output_dir = Config.OUTPUT_DIR
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    plot_file = os.path.join(output_dir, "generalization_comparison.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"\nGeneralization comparison saved to: {plot_file}")
    
    # 计算相对改进
    print("\nPerformance Improvements:")
    if len(accuracies) >= 3:
        zero_shot_acc = accuracies[0]
        few_shot_acc = accuracies[1]
        full_finetune_acc = accuracies[2]
        
        print(f"  Zero-shot → Few-shot: +{(few_shot_acc - zero_shot_acc)*100:.2f}%")
        print(f"  Few-shot → Full Fine-tuning: +{(full_finetune_acc - few_shot_acc)*100:.2f}%")
        print(f"  Zero-shot → Full Fine-tuning: +{(full_finetune_acc - zero_shot_acc)*100:.2f}%")
    
    plt.show()


def run_generalization_experiments():
    """
    运行完整的泛化实验
    """
    print("\n" + "=" * 60)
    print("GENERALIZATION EXPERIMENTS")
    print("=" * 60)
    
    # 加载数据
    from .dataset import TextClassificationDataset
    data_handler = TextClassificationDataset()
    train_dataset, val_dataset, test_dataset, tokenizer = data_handler.load_and_preprocess()
    
    all_results = {}
    
    try:
        print("\n1. Running Zero-shot Experiment...")
        zero_shot_result = run_zero_shot_experiment(
            test_dataset.select(range(500)),  # 使用部分测试数据
            tokenizer
        )
        all_results["zero_shot"] = zero_shot_result
    except Exception as e:
        print(f"Zero-shot experiment failed: {e}")
    
    try:
        print("\n2. Running Few-shot Experiments...")
        few_shot_results = run_few_shot_experiment(
            train_dataset.select(range(2000)),  # 使用部分训练数据
            val_dataset.select(range(200)),
            test_dataset.select(range(500)),
            tokenizer,
            n_shots_list=[1, 2, 5, 10, 50, 100]
        )
        all_results["few_shot"] = few_shot_results
    except Exception as e:
        print(f"Few-shot experiment failed: {e}")
    
    try:
        print("\n3. Running Scaling Analysis...")
        scaling_results = run_scaling_analysis(
            train_dataset.select(range(10000)),  # 使用更多数据用于缩放分析
            val_dataset.select(range(200)),
            test_dataset.select(range(500)),
            tokenizer,
            sample_sizes=[200, 500, 1000, 2000, 5000, 20000]
        )
        all_results["scaling"] = scaling_results
    except Exception as e:
        print(f"Scaling analysis failed: {e}")
    
    # 比较结果
    if all_results:
        compare_generalization_methods(
            all_results.get("zero_shot", {}),
            all_results.get("few_shot", []),
            all_results.get("scaling", [])
        )
    
    print("\n" + "=" * 60)
    print("GENERALIZATION EXPERIMENTS COMPLETED")
    print("=" * 60)
    
    return all_results


if __name__ == "__main__":
    run_generalization_experiments()