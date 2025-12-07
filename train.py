import torch
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
from transformers import EarlyStoppingCallback
from .config import Config
from .dataset import TextClassificationDataset
from .model import get_model
from .utils import compute_metrics
import numpy as np
from sklearn.metrics import classification_report
import random
import time


def run_baseline_training(seed=None, subset_size=None):
    """
    运行基线模型训练 (不使用 LoRA，只用标准 DistilBERT)
    
    Args:
        seed: 随机种子
        subset_size: 训练集子集大小，None表示使用全部数据
    
    Returns:
        包含训练结果的字典
    """
    # 使用指定的种子或配置中的种子
    current_seed = seed if seed is not None else Config.SEED
    print("=" * 60)
    print("BASELINE MODEL TRAINING (Standard DistilBERT)")
    print("=" * 60)
    print(f"Using seed: {current_seed}")
    
    # 设置随机种子以确保可重复性
    torch.manual_seed(current_seed)
    np.random.seed(current_seed)
    random.seed(current_seed)
    
    # 1. Prepare Data
    print("\n[1/5] Loading and preprocessing data...")
    data_handler = TextClassificationDataset()
    train_dataset, val_dataset, test_dataset, tokenizer = (
        data_handler.load_and_preprocess()
    )

    # 如果指定了子集大小，从训练集中随机抽取子集
    if subset_size is not None and subset_size < len(train_dataset):
        print(f"\nUsing subset of training data: {subset_size} samples out of {len(train_dataset)}")
        
        # 随机选择索引
        subset_indices = random.sample(range(len(train_dataset)), subset_size)
        train_dataset = train_dataset.select(subset_indices)
        
        # 也减小验证集大小以保持比例
        val_subset_size = max(100, int(subset_size * 0.1))
        val_indices = random.sample(range(len(val_dataset)), min(val_subset_size, len(val_dataset)))
        val_dataset = val_dataset.select(val_indices)

        print(f"Training subset size: {len(train_dataset)}")
        print(f"Validation subset size: {len(val_dataset)}")
    else:
        print(f"Using full dataset")
    
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    print(f"Test set size: {len(test_dataset)}")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 2. Load Baseline Model (WITHOUT LoRA)
    print("\n[2/5] Loading baseline model (no LoRA)...")
    model = get_model(use_lora=False)  # ✅ 关键: 不使用 LoRA
    
    # 打印模型参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Trainable %: {100 * trainable_params / total_params:.2f}%")

    # 3. Zero-shot Evaluation (Optional - 快速检查)
    print("\n[3/5] Running zero-shot evaluation (before training)...")
    eval_args = TrainingArguments(
        output_dir=Config.OUTPUT_DIR,
        per_device_eval_batch_size=Config.BATCH_SIZE,
        seed=current_seed,
    )

    zero_shot_trainer = Trainer(
        model=model,
        args=eval_args,
        eval_dataset=test_dataset.select(range(min(100, len(test_dataset)))),
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    zero_shot_metrics = zero_shot_trainer.evaluate()
    print(f"Zero-shot accuracy: {zero_shot_metrics.get('eval_accuracy', 0):.4f}")
    print(f"Zero-shot F1: {zero_shot_metrics.get('eval_f1', 0):.4f}")

    # 4. Setup Training Arguments
    print("\n[4/5] Setting up training configuration...")
    training_args = TrainingArguments(
        output_dir=f"{Config.OUTPUT_DIR}/baseline_seed_{current_seed}",
        learning_rate=Config.LEARNING_RATE,
        per_device_train_batch_size=Config.BATCH_SIZE,
        per_device_eval_batch_size=Config.BATCH_SIZE,
        num_train_epochs=Config.NUM_EPOCHS,
        weight_decay=Config.WEIGHT_DECAY,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        logging_dir=f"{Config.LOG_DIR}/baseline_seed_{current_seed}",
        logging_steps=10,
        report_to="none",
        seed=current_seed,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # 5. Train
    print("\n[5/5] Starting training...")
    print(f"Training for {Config.NUM_EPOCHS} epochs...")
    print("-" * 60)
    
    start_time = time.time()
    train_result = trainer.train()
    training_time = time.time() - start_time
    
    # 记录训练指标
    metrics = train_result.metrics
    print(f"\nTraining completed in {training_time:.2f} seconds")
    print(f"Training loss: {metrics.get('train_loss', 'N/A'):.4f}")

    # 6. Final Evaluation on Test Set
    print("\n" + "=" * 60)
    print("FINAL EVALUATION ON TEST SET")
    print("=" * 60)
    test_results = trainer.evaluate(test_dataset)
    
    print(f"\nTest Accuracy: {test_results['eval_accuracy']:.4f}")
    print(f"Test F1 Score: {test_results['eval_f1']:.4f}")
    print(f"Test Loss: {test_results['eval_loss']:.4f}")

    # 7. Detailed Classification Report
    print("\n" + "=" * 60)
    print("DETAILED CLASSIFICATION REPORT")
    print("=" * 60)
    predictions = trainer.predict(test_dataset)
    
    if predictions and hasattr(predictions, 'predictions'):
        preds = np.argmax(predictions.predictions, axis=-1)
        labels = test_dataset["labels"]
        
        print("\nClassification Report:")
        print(classification_report(
            labels, 
            preds,
            target_names=["World", "Sports", "Business", "Sci/Tech"],
            digits=4
        ))
        
        # 完整的测试结果可视化
        from .utils import plot_test_results
        
        plot_test_results(
            test_results=test_results,
            predictions=preds,
            labels=labels,
            class_names=["World", "Sports", "Business", "Sci/Tech"],
            model_name="Baseline DistilBERT",
            output_dir=f"{Config.OUTPUT_DIR}/baseline_seed_{current_seed}",
            seed=current_seed
        )
        
        # 绘制混淆矩阵
        try:
            from .utils import plot_confusion_matrix
            plot_confusion_matrix(
                preds, 
                labels, 
                class_names=["World", "Sports", "Business", "Sci/Tech"],
                output_dir=f"{Config.OUTPUT_DIR}/baseline_seed_{current_seed}",
                title="Baseline Model Confusion Matrix"
            )
            print(f"\nConfusion matrix saved to {Config.OUTPUT_DIR}/baseline_seed_{current_seed}")
        except Exception as e:
            print(f"Warning: Could not plot confusion matrix: {e}")

    # 8. Save Model
    save_path = f"{Config.OUTPUT_DIR}/baseline_model_seed_{current_seed}"
    trainer.save_model(save_path)
    print(f"\n✅ Baseline model saved to {save_path}")
    
    # 9. Summary
    print("\n" + "=" * 60)
    print("BASELINE TRAINING SUMMARY")
    print("=" * 60)
    print(f"Model: {Config.MODEL_NAME} (Standard, no LoRA)")
    print(f"Seed: {current_seed}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Final test accuracy: {test_results['eval_accuracy']:.4f}")
    print(f"Final test F1: {test_results['eval_f1']:.4f}")
    print("=" * 60)
    
    # 返回测试结果
    return {
        "seed": current_seed,
        "accuracy": test_results["eval_accuracy"],
        "f1": test_results["eval_f1"],
        "loss": test_results["eval_loss"],
        "training_time": training_time,
        "model_type": "baseline",
        "use_lora": False,
    }


def run_training(seed=None, use_early_stopping=True, subset_size=None, use_lora=True):
    """
    运行训练 (可选择使用 LoRA 或标准模型)
    
    Args:
        seed: 随机种子
        use_early_stopping: 是否使用早停
        subset_size: 训练集子集大小
        use_lora: 是否使用 LoRA (False = baseline model)
    """
    if not use_lora:
        # 如果不使用 LoRA，调用 baseline 训练函数
        return run_baseline_training(seed=seed, subset_size=subset_size)
    
    # 原有的 LoRA 训练代码...
    # 使用指定的种子或配置中的种子
    current_seed = seed if seed is not None else Config.SEED
    print(f"Using seed: {current_seed}")
    print(f"Using LoRA: {use_lora}")
    
    # 设置随机种子以确保可重复性
    torch.manual_seed(current_seed)
    np.random.seed(current_seed)
    
    # 1. Prepare Data
    print("\n[1/5] Loading and preprocessing data...")
    data_handler = TextClassificationDataset()
    train_dataset, val_dataset, test_dataset, tokenizer = (
        data_handler.load_and_preprocess()
    )

    # 如果指定了子集大小，从训练集中随机抽取子集
    if subset_size is not None and subset_size < len(train_dataset):
        print(f"\nUsing subset of training data: {subset_size} samples out of {len(train_dataset)}")
        
        # 随机选择索引
        subset_indices = random.sample(range(len(train_dataset)), subset_size)
        train_dataset = train_dataset.select(subset_indices)
        
        # 从子集中重新划分验证集
        val_subset_size = max(100, int(subset_size * 0.1))
        val_indices = random.sample(range(len(val_dataset)), min(val_subset_size, len(val_dataset)))
        val_dataset = val_dataset.select(val_indices)
    
        print(f"Training subset size: {len(train_dataset)}")
        print(f"Validation subset size: {len(val_dataset)}")
    else:
        print(f"Using full dataset")

    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    print(f"Test set size: {len(test_dataset)}")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 2. Load Model
    print("\n[2/5] Loading LoRA model...")
    model = get_model(use_lora=use_lora)

    # 3. Zero-shot Evaluation
    print("\n[3/5] Running zero-shot evaluation (before training)...")
    eval_args = TrainingArguments(
        output_dir=Config.OUTPUT_DIR,
        per_device_eval_batch_size=Config.BATCH_SIZE,
        seed=current_seed,
    )

    zero_shot_trainer = Trainer(
        model=model,
        args=eval_args,
        eval_dataset=test_dataset.select(range(100)),
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    zero_shot_metrics = zero_shot_trainer.evaluate()
    print(f"Zero-shot Metrics: {zero_shot_metrics}")

    # 4. Setup Training with Early Stopping
    print("\n[4/5] Setting up training configuration...")
    callbacks = []
    if use_early_stopping:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=3,
                early_stopping_threshold=0.001,
            )
        )
        print("Early stopping enabled with patience=3")

    model_prefix = "lora" if use_lora else "baseline"
    training_args = TrainingArguments(
        output_dir=f"{Config.OUTPUT_DIR}/{model_prefix}_seed_{current_seed}",
        learning_rate=Config.LEARNING_RATE,
        per_device_train_batch_size=Config.BATCH_SIZE,
        per_device_eval_batch_size=Config.BATCH_SIZE,
        num_train_epochs=Config.NUM_EPOCHS,
        weight_decay=Config.WEIGHT_DECAY,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        logging_dir=f"{Config.LOG_DIR}/{model_prefix}_seed_{current_seed}",
        logging_steps=10,
        report_to="none",
        seed=current_seed,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    # 5. Train
    print("\n[5/5] Starting training...")
    print(f"Training for {Config.NUM_EPOCHS} epochs...")
    print("-" * 60)

    start_time = time.time()
    train_result = trainer.train()
    training_time = time.time() - start_time
    
    # 记录训练指标
    metrics = train_result.metrics
    print(f"Training completed. Metrics: {metrics}")

    # 6. Final Evaluation
    print("\n--- Running Final Evaluation on Test Set ---")
    test_results = trainer.evaluate(test_dataset)
    print(f"Test Results: {test_results}")

    # 7. Detailed Classification Report and Confusion Matrix
    predictions = trainer.predict(test_dataset)
    
    if predictions and hasattr(predictions, 'predictions'):
        preds = np.argmax(predictions.predictions, axis=-1)
        
        print("\nClassification Report:")
        print(classification_report(test_dataset["labels"], preds))

        # 完整的测试结果可视化
        from .utils import plot_test_results
        
        model_name = "LoRA DistilBERT" if use_lora else "Standard DistilBERT"
        plot_test_results(
            test_results=test_results,
            predictions=preds,
            labels=test_dataset["labels"],
            class_names=["World", "Sports", "Business", "Sci/Tech"],
            model_name=model_name,
            output_dir=f"{Config.OUTPUT_DIR}/{model_prefix}_seed_{current_seed}",
            seed=current_seed
        )
        
        # 绘制混淆矩阵
        try:
            from .utils import plot_confusion_matrix
            plot_confusion_matrix(
                preds, 
                test_dataset["labels"], 
                class_names=["World", "Sports", "Business", "Sci/Tech"],
                output_dir=f"{Config.OUTPUT_DIR}/{model_prefix}_seed_{current_seed}"
            )
        except Exception as e:
            print(f"Warning: Could not plot confusion matrix: {e}")

    # 8. Save Model
    trainer.save_model(f"{Config.OUTPUT_DIR}/{model_prefix}_seed_{current_seed}")
    print(f"Model saved to {Config.OUTPUT_DIR}/{model_prefix}_seed_{current_seed}")
    
    # 9. Return Test Results
    return {
        "seed": current_seed,
        "accuracy": test_results["eval_accuracy"],
        "f1": test_results["eval_f1"],
        "loss": test_results["eval_loss"],
        "training_time": training_time,
        "use_lora": use_lora,
    }

def run_short_training(train_dataset, val_dataset, test_dataset=None, tokenizer=None, 
                       num_epochs=2, use_lora=True, learning_rate=None, weight_decay=None):
    """
    简短的训练函数，用于实验
    """
    from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
    from .model import get_model
    from .utils import compute_metrics
    
    if test_dataset is None:
        test_dataset = val_dataset
    
    # 设置随机种子
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # 加载模型
    model = get_model(use_lora=use_lora)
    
    # 配置训练参数
    training_args = TrainingArguments(
        output_dir=f"{Config.OUTPUT_DIR}/experiment_temp",
        learning_rate=learning_rate if learning_rate is not None else Config.LEARNING_RATE,
        weight_decay=weight_decay if weight_decay is not None else Config.WEIGHT_DECAY,
        per_device_train_batch_size=Config.BATCH_SIZE,
        per_device_eval_batch_size=Config.BATCH_SIZE,
        num_train_epochs=num_epochs,
        eval_strategy="epoch",
        save_strategy="no",
        logging_dir=f"{Config.LOG_DIR}/experiment_temp",
        logging_steps=5,
        report_to="none",
        seed=Config.SEED,
        disable_tqdm=True,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # 训练
    train_result = trainer.train()
    
    # 在测试集上评估
    test_results = trainer.evaluate(test_dataset)
    
    return {
        'accuracy': test_results["eval_accuracy"],
        'f1': test_results["eval_f1"],
        'loss': test_results["eval_loss"],
        'training_time': train_result.metrics.get("train_runtime", 0),
    }



if __name__ == "__main__":
    # 先运行 baseline
    print("Running BASELINE model training...")
    baseline_results = run_baseline_training()
    
    print("\n\n")
    print("Running LoRA model training...")
    lora_results = run_training(use_lora=True, subset_size=2400)
    
    # 比较结果
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    print(f"Baseline - Accuracy: {baseline_results['accuracy']:.4f}, F1: {baseline_results['f1']:.4f}")
    print(f"LoRA     - Accuracy: {lora_results['accuracy']:.4f}, F1: {lora_results['f1']:.4f}")
    print("=" * 60)
