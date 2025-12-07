import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .config import Config
import os
from datetime import datetime


class HyperparameterSweep:
    def __init__(self):
        self.results = []
        self.best_config = None
        
    def run_learning_rate_sweep(self, learning_rates=None, num_epochs=2):
        """
        运行学习率扫描实验
        """
        if learning_rates is None:
            learning_rates = [1e-5, 5e-5, 1e-4, 2e-4, 5e-4]
        
        print("=" * 60)
        print("LEARNING RATE SWEEP EXPERIMENT")
        print("=" * 60)
        
        original_lr = Config.LEARNING_RATE
        
        for lr in learning_rates:
            print(f"\nTraining with learning_rate = {lr}")
            print("-" * 40)
            
            # 临时修改配置
            Config.LEARNING_RATE = lr
            
            try:
                # 运行训练（减少epoch数以加快扫描速度）
                result = self.run_short_training(lr=lr, num_epochs=num_epochs)
                self.results.append(result)
                print(f"Results - Accuracy: {result['accuracy']:.4f}, F1: {result['f1']:.4f}")
                
            except Exception as e:
                print(f"Training failed with lr={lr}: {e}")
            
            finally:
                # 恢复原始学习率
                Config.LEARNING_RATE = original_lr
        
        self.analyze_results()
        self.plot_lr_sweep()
        
        return self.results
    
    def run_weight_decay_sweep(self, weight_decays=None, num_epochs=2):
        """
        运行权重衰减扫描实验
        """
        if weight_decays is None:
            weight_decays = [0.0, 0.001, 0.01, 0.05, 0.1]
        
        print("=" * 60)
        print("WEIGHT DECAY SWEEP EXPERIMENT")
        print("=" * 60)
        
        original_wd = Config.WEIGHT_DECAY
        
        for wd in weight_decays:
            print(f"\nTraining with weight_decay = {wd}")
            print("-" * 40)
            
            # 临时修改配置
            Config.WEIGHT_DECAY = wd
            
            try:
                # 运行训练
                result = self.run_short_training(weight_decay=wd, num_epochs=num_epochs)
                self.results.append(result)
                print(f"Results - Accuracy: {result['accuracy']:.4f}, F1: {result['f1']:.4f}")
                
            except Exception as e:
                print(f"Training failed with wd={wd}: {e}")
            
            finally:
                # 恢复原始权重衰减
                Config.WEIGHT_DECAY = original_wd
        
        self.analyze_results()
        self.plot_weight_decay_sweep()
        
        return self.results
    
    def run_combined_sweep(self, learning_rates=None, weight_decays=None, num_epochs=1):
        """
        运行组合超参数扫描（学习率 × 权重衰减）
        """
        if learning_rates is None:
            learning_rates = [1e-5, 5e-5, 1e-4, 2e-4, 5e-4]
        
        if weight_decays is None:
            weight_decays = [0.0, 0.001, 0.01, 0.05, 0.1]
        
        print("=" * 60)
        print("COMBINED HYPERPARAMETER SWEEP")
        print("=" * 60)
        
        original_lr = Config.LEARNING_RATE
        original_wd = Config.WEIGHT_DECAY
        
        param_combinations = list(itertools.product(learning_rates, weight_decays))
        print(f"Testing {len(param_combinations)} parameter combinations")
        
        for i, (lr, wd) in enumerate(param_combinations, 1):
            print(f"\n[{i}/{len(param_combinations)}] lr={lr}, wd={wd}")
            print("-" * 40)
            
            # 临时修改配置
            Config.LEARNING_RATE = lr
            Config.WEIGHT_DECAY = wd
            
            try:
                # 运行训练
                result = self.run_short_training(lr=lr, weight_decay=wd, num_epochs=num_epochs)
                result['lr'] = lr
                result['wd'] = wd
                self.results.append(result)
                print(f"Results - Accuracy: {result['accuracy']:.4f}, F1: {result['f1']:.4f}")
                
            except Exception as e:
                print(f"Training failed with lr={lr}, wd={wd}: {e}")
            
            finally:
                # 恢复原始配置
                Config.LEARNING_RATE = original_lr
                Config.WEIGHT_DECAY = original_wd
        
        self.analyze_combined_results()
        self.plot_combined_sweep()
        
        return self.results
    
    def run_short_training(self, lr=None, weight_decay=None, num_epochs=2):
        """
        运行简短训练以进行超参数扫描
        """
        from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
        from transformers import EarlyStoppingCallback
        from .dataset import TextClassificationDataset
        from .model import get_model
        from .utils import compute_metrics
        
        # 设置随机种子
        seed = Config.SEED
        
        # 准备数据
        data_handler = TextClassificationDataset()
        train_dataset, val_dataset, test_dataset, tokenizer = data_handler.load_and_preprocess()
        
        # 使用小部分数据以加快扫描速度
        small_train = train_dataset.select(range(1000))
        small_val = val_dataset.select(range(200))
        small_test = test_dataset.select(range(200))
        
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        
        # 加载模型
        model = get_model(use_lora=True)
        
        # 配置训练参数
        training_args = TrainingArguments(
            output_dir=f"{Config.OUTPUT_DIR}/sweep_temp",
            learning_rate=lr if lr is not None else Config.LEARNING_RATE,
            weight_decay=weight_decay if weight_decay is not None else Config.WEIGHT_DECAY,
            per_device_train_batch_size=Config.BATCH_SIZE,
            per_device_eval_batch_size=Config.BATCH_SIZE,
            num_train_epochs=num_epochs,
            eval_strategy="epoch",
            save_strategy="no",
            logging_dir=f"{Config.LOG_DIR}/sweep_temp",
            logging_steps=5,
            report_to="none",
            seed=seed,
            disable_tqdm=True,  # 禁用进度条以简化输出
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=small_train,
            eval_dataset=small_val,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
        
        # 训练
        train_result = trainer.train()
        
        # 在测试集上评估
        test_results = trainer.evaluate(small_test)
        
        return {
            "learning_rate": lr if lr is not None else Config.LEARNING_RATE,
            "weight_decay": weight_decay if weight_decay is not None else Config.WEIGHT_DECAY,
            "accuracy": test_results["eval_accuracy"],
            "f1": test_results["eval_f1"],
            "loss": test_results["eval_loss"],
            "training_time": train_result.metrics.get("train_runtime", 0),
        }
    
    def analyze_results(self):
        """
        分析扫描结果
        """
        if not self.results:
            print("No results to analyze!")
            return
        
        df = pd.DataFrame(self.results)
        
        # 找到最佳配置
        best_idx = df['accuracy'].idxmax()
        self.best_config = df.loc[best_idx].to_dict()
        
        print("\n" + "=" * 60)
        print("ANALYSIS RESULTS")
        print("=" * 60)
        
        print(f"\nBest configuration:")
        for key, value in self.best_config.items():
            print(f"  {key}: {value}")
        
        print(f"\nTotal configurations tested: {len(df)}")
        print(f"Accuracy range: [{df['accuracy'].min():.4f}, {df['accuracy'].max():.4f}]")
        print(f"F1 range: [{df['f1'].min():.4f}, {df['f1'].max():.4f}]")
    
    def analyze_combined_results(self):
        """
        分析组合扫描结果
        """
        if not self.results:
            print("No results to analyze!")
            return
        
        df = pd.DataFrame(self.results)
        
        # 找到最佳配置
        best_idx = df['accuracy'].idxmax()
        self.best_config = df.loc[best_idx].to_dict()
        
        print("\n" + "=" * 60)
        print("COMBINED ANALYSIS RESULTS")
        print("=" * 60)
        
        # 创建透视表
        pivot_acc = df.pivot(index='lr', columns='wd', values='accuracy')
        pivot_f1 = df.pivot(index='lr', columns='wd', values='f1')
        
        print("\nAccuracy matrix (lr × wd):")
        print(pivot_acc.round(4))
        
        print("\nF1 matrix (lr × wd):")
        print(pivot_f1.round(4))
        
        print(f"\nBest configuration:")
        for key, value in self.best_config.items():
            print(f"  {key}: {value}")
    
    def plot_lr_sweep(self):
        """
        绘制学习率扫描结果
        """
        if not self.results:
            return
        
        df = pd.DataFrame(self.results)
        df = df.sort_values('learning_rate')
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. 准确率 vs 学习率
        axes[0, 0].plot(df['learning_rate'], df['accuracy'], 'o-', linewidth=2, markersize=8)
        axes[0, 0].set_xscale('log')
        axes[0, 0].set_xlabel('Learning Rate (log scale)')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_title('Accuracy vs Learning Rate')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. F1 vs 学习率
        axes[0, 1].plot(df['learning_rate'], df['f1'], 's-', linewidth=2, markersize=8, color='orange')
        axes[0, 1].set_xscale('log')
        axes[0, 1].set_xlabel('Learning Rate (log scale)')
        axes[0, 1].set_ylabel('F1 Score')
        axes[0, 1].set_title('F1 Score vs Learning Rate')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 损失 vs 学习率
        axes[1, 0].plot(df['learning_rate'], df['loss'], '^-', linewidth=2, markersize=8, color='red')
        axes[1, 0].set_xscale('log')
        axes[1, 0].set_xlabel('Learning Rate (log scale)')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title('Loss vs Learning Rate')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 训练时间 vs 学习率
        axes[1, 1].plot(df['learning_rate'], df['training_time'], 'd-', linewidth=2, markersize=8, color='green')
        axes[1, 1].set_xscale('log')
        axes[1, 1].set_xlabel('Learning Rate (log scale)')
        axes[1, 1].set_ylabel('Training Time (seconds)')
        axes[1, 1].set_title('Training Time vs Learning Rate')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图像
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Config.OUTPUT_DIR
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        plot_file = os.path.join(output_dir, f"lr_sweep_{timestamp}.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"\nLearning rate sweep plot saved to: {plot_file}")
        
        # 保存数据
        data_file = os.path.join(output_dir, f"lr_sweep_results_{timestamp}.csv")
        df.to_csv(data_file, index=False)
        print(f"Learning rate sweep data saved to: {data_file}")
        
        plt.show()
    
    def plot_weight_decay_sweep(self):
        """
        绘制权重衰减扫描结果
        """
        if not self.results:
            return
        
        df = pd.DataFrame(self.results)
        df = df.sort_values('weight_decay')
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. 准确率 vs 权重衰减
        axes[0, 0].plot(df['weight_decay'], df['accuracy'], 'o-', linewidth=2, markersize=8)
        axes[0, 0].set_xlabel('Weight Decay')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_title('Accuracy vs Weight Decay')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. F1 vs 权重衰减
        axes[0, 1].plot(df['weight_decay'], df['f1'], 's-', linewidth=2, markersize=8, color='orange')
        axes[0, 1].set_xlabel('Weight Decay')
        axes[0, 1].set_ylabel('F1 Score')
        axes[0, 1].set_title('F1 Score vs Weight Decay')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 损失 vs 权重衰减
        axes[1, 0].plot(df['weight_decay'], df['loss'], '^-', linewidth=2, markersize=8, color='red')
        axes[1, 0].set_xlabel('Weight Decay')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title('Loss vs Weight Decay')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 准确率和F1对比
        axes[1, 1].plot(df['weight_decay'], df['accuracy'], 'o-', linewidth=2, markersize=8, label='Accuracy')
        axes[1, 1].plot(df['weight_decay'], df['f1'], 's-', linewidth=2, markersize=8, label='F1 Score')
        axes[1, 1].set_xlabel('Weight Decay')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_title('Accuracy and F1 vs Weight Decay')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图像
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Config.OUTPUT_DIR
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        plot_file = os.path.join(output_dir, f"wd_sweep_{timestamp}.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"\nWeight decay sweep plot saved to: {plot_file}")
        
        # 保存数据
        data_file = os.path.join(output_dir, f"wd_sweep_results_{timestamp}.csv")
        df.to_csv(data_file, index=False)
        print(f"Weight decay sweep data saved to: {data_file}")
        
        plt.show()
    
    def plot_combined_sweep(self):
        """
        绘制组合超参数扫描结果
        """
        if not self.results:
            return
        
        df = pd.DataFrame(self.results)
        
        # 创建透视表
        pivot_acc = df.pivot(index='lr', columns='wd', values='accuracy')
        pivot_f1 = df.pivot(index='lr', columns='wd', values='f1')
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 1. 准确率热力图
        im1 = axes[0].imshow(pivot_acc.values, cmap='viridis', aspect='auto')
        axes[0].set_title('Accuracy Heatmap')
        axes[0].set_xlabel('Weight Decay')
        axes[0].set_ylabel('Learning Rate')
        
        # 设置刻度标签
        axes[0].set_xticks(range(len(pivot_acc.columns)))
        axes[0].set_xticklabels([f'{x:.3f}' for x in pivot_acc.columns])
        axes[0].set_yticks(range(len(pivot_acc.index)))
        axes[0].set_yticklabels([f'{x:.0e}' for x in pivot_acc.index])
        
        # 添加数值标签
        for i in range(len(pivot_acc.index)):
            for j in range(len(pivot_acc.columns)):
                axes[0].text(j, i, f'{pivot_acc.iloc[i, j]:.3f}', 
                           ha='center', va='center', color='white')
        
        plt.colorbar(im1, ax=axes[0])
        
        # 2. F1热力图
        im2 = axes[1].imshow(pivot_f1.values, cmap='plasma', aspect='auto')
        axes[1].set_title('F1 Score Heatmap')
        axes[1].set_xlabel('Weight Decay')
        axes[1].set_ylabel('Learning Rate')
        
        axes[1].set_xticks(range(len(pivot_f1.columns)))
        axes[1].set_xticklabels([f'{x:.3f}' for x in pivot_f1.columns])
        axes[1].set_yticks(range(len(pivot_f1.index)))
        axes[1].set_yticklabels([f'{x:.0e}' for x in pivot_f1.index])
        
        for i in range(len(pivot_f1.index)):
            for j in range(len(pivot_f1.columns)):
                axes[1].text(j, i, f'{pivot_f1.iloc[i, j]:.3f}', 
                           ha='center', va='center', color='white')
        
        plt.colorbar(im2, ax=axes[1])
        
        # 3. 3D散点图
        ax3d = fig.add_subplot(133, projection='3d')
        
        # 转换为对数尺度以便可视化
        lr_log = np.log10(df['lr'])
        
        scatter = ax3d.scatter(lr_log, df['wd'], df['accuracy'], 
                              c=df['f1'], cmap='coolwarm', s=100)
        
        ax3d.set_xlabel('log10(Learning Rate)')
        ax3d.set_ylabel('Weight Decay')
        ax3d.set_zlabel('Accuracy')
        ax3d.set_title('3D: LR vs WD vs Accuracy (color=F1)')
        
        plt.colorbar(scatter, ax=ax3d, label='F1 Score')
        
        plt.tight_layout()
        
        # 保存图像
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Config.OUTPUT_DIR
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        plot_file = os.path.join(output_dir, f"combined_sweep_{timestamp}.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"\nCombined sweep plot saved to: {plot_file}")
        
        # 保存数据
        data_file = os.path.join(output_dir, f"combined_sweep_results_{timestamp}.csv")
        df.to_csv(data_file, index=False)
        print(f"Combined sweep data saved to: {data_file}")
        
        plt.show()


def run_hyperparameter_experiments():
    """
    运行完整的超参数实验
    """
    sweep = HyperparameterSweep()
    
    print("Starting hyperparameter experiments...")
    print("=" * 60)
    
    # 1. 学习率扫描
    print("\n[1/3] Running learning rate sweep...")
    lr_results = sweep.run_learning_rate_sweep(
        learning_rates=[1e-5, 5e-5, 1e-4, 2e-4, 5e-4],
        num_epochs=2
    )
    
    # 重置结果
    sweep.results = []
    
    # 2. 权重衰减扫描
    print("\n[2/3] Running weight decay sweep...")
    wd_results = sweep.run_weight_decay_sweep(
        weight_decays=[0.0, 0.001, 0.01, 0.05, 0.1],
        num_epochs=2
    )
    
    # 重置结果
    sweep.results = []
    
    # 3. 组合扫描
    print("\n[3/3] Running combined hyperparameter sweep...")
    combined_results = sweep.run_combined_sweep(
        learning_rates=[1e-5, 5e-5, 1e-4, 2e-4, 5e-4],
        weight_decays=[0.0, 0.001, 0.01, 0.05, 0.1],
        num_epochs=1
    )
    
    print("\n" + "=" * 60)
    print("HYPERPARAMETER EXPERIMENTS COMPLETED")
    print("=" * 60)
    
    # 总结最佳配置
    if sweep.best_config:
        print("\nRECOMMENDED HYPERPARAMETERS:")
        print(f"  Learning Rate: {sweep.best_config.get('learning_rate', sweep.best_config.get('lr', 'N/A'))}")
        print(f"  Weight Decay: {sweep.best_config.get('weight_decay', sweep.best_config.get('wd', 'N/A'))}")
        print(f"  Expected Accuracy: {sweep.best_config.get('accuracy', 'N/A'):.4f}")
        print(f"  Expected F1: {sweep.best_config.get('f1', 'N/A'):.4f}")
    
    return sweep.best_config


if __name__ == "__main__":
    run_hyperparameter_experiments()