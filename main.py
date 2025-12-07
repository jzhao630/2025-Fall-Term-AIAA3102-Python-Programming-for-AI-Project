import argparse
from src.train import run_training
from src.multi_seed_experiment import run_multi_seed_experiment
from src.hyperparameter_sweep import run_hyperparameter_experiments
from src.data_centric import run_data_centric_experiments
from src.generalization import run_generalization_experiments


def main():
    parser = argparse.ArgumentParser(
        description="Project A: Fine-Tune a Small Language Model"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "compare", "multi_seed", "hyperparam", 
                 "data_centric", "generalization", "clean_test"],
        help="Mode: 'train' for full pipeline, 'compare' for efficiency comparison, "
             "'multi_seed' for multi-seed experiment, 'hyperparam' for hyperparameter sweep, "
             "'data_centric' for data-centric experiments, 'generalization' for generalization experiments"
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Enable text cleaning (for train mode)"
    )
    parser.add_argument(
        "--early_stop",
        action="store_true",
        default=True,
        help="Enable early stopping (default: True)"
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=5,
        help="Number of runs for multi-seed experiment (default: 5)"
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="",
        help="Comma-separated list of seeds for multi-seed experiment (e.g., '42,123,456')"
    )
    parser.add_argument(
        "--sweep_type",
        type=str,
        default="all",
        choices=["lr", "wd", "combined", "all"],
        help="Type of hyperparameter sweep: 'lr' for learning rate, 'wd' for weight decay, "
             "'combined' for both, 'all' for all sweeps"
    )

    args = parser.parse_args()

    if args.mode == "train":
        print("Running BASELINE model training...")
        baseline_results = run_training(use_lora=False, subset_size=2400)

        print("Running LoRA model training...")
        lora_results = run_training(use_lora=True, subset_size=2400)

        # 比较结果
        print("\n" + "=" * 60)
        print("MODEL COMPARISON")
        print("=" * 60)
        print(f"Baseline - Accuracy: {baseline_results['accuracy']:.4f}, F1: {baseline_results['f1']:.4f}")
        print(f"LoRA     - Accuracy: {lora_results['accuracy']:.4f}, F1: {lora_results['f1']:.4f}")
        print("=" * 60)

    elif args.mode == "multi_seed":
        # 解析种子列表
        if args.seeds:
            seeds = [int(s.strip()) for s in args.seeds.split(",")]
        else:
            seeds = None
        run_multi_seed_experiment(seeds=seeds, num_runs=args.num_runs)

    elif args.mode == "hyperparam":
        print("Running hyperparameter sweep experiments...")
        if args.sweep_type == "all":
            run_hyperparameter_experiments()
        else:
            from src.hyperparameter_sweep import HyperparameterSweep
            sweep = HyperparameterSweep()
            
            if args.sweep_type == "lr":
                sweep.run_learning_rate_sweep(
                    learning_rates=[1e-5, 5e-5, 1e-4, 2e-4, 5e-4],
                    num_epochs=2
                )
            elif args.sweep_type == "wd":
                sweep.run_weight_decay_sweep(
                    weight_decays=[0.0, 0.001, 0.01, 0.05, 0.1],
                    num_epochs=2
                )
            elif args.sweep_type == "combined":
                sweep.run_combined_sweep(
                    learning_rates=[1e-5, 5e-5, 1e-4, 2e-4, 5e-4],
                    weight_decays=[0.0, 0.001, 0.01, 0.05, 0.1],
                    num_epochs=1
                )
    elif args.mode == "data_centric":
        print("Running data-centric experiments...")
        run_data_centric_experiments()

    elif args.mode == "generalization":
        print("Running generalization experiments...")
        run_generalization_experiments()

    elif args.mode == "clean_test":
        # 测试数据清洗
        from src.dataset import TextClassificationDataset
        handler = TextClassificationDataset()
        
        # 测试清洗函数
        test_texts = [
            "Check out this link: http://example.com <b>bold text</b>!!!",
            "HTML entities: &amp; &lt; &gt;",
            "Multiple   spaces   and\nnewlines"
        ]
        
        print("Testing text cleaning function:")
        print("-" * 40)
        for text in test_texts:
            cleaned = handler.clean_text(text)
            print(f"Original: {text}")
            print(f"Cleaned:  {cleaned}")
            print("-" * 40)


if __name__ == "__main__":
    main()