class Config:
    # Model and Data
    MODEL_NAME = "distilbert-base-uncased"
    DATASET_NAME = "ag_news"
    NUM_LABELS = 4

    # Training Hyperparameters
    LEARNING_RATE = 5e-4
    BATCH_SIZE = 16
    NUM_EPOCHS = 3
    WEIGHT_DECAY = 0.01

    # LoRA Hyperparameters
    LORA_R = 8
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.1

    # Early Stopping
    EARLY_STOPPING_PATIENCE = 3
    EARLY_STOPPING_THRESHOLD = 0.001

    # Hyperparameter Sweep Ranges
    LR_SWEEP_RANGE = [1e-5, 5e-5, 1e-4, 2e-4, 5e-4]
    WD_SWEEP_RANGE = [0.0, 0.001, 0.01, 0.05, 0.1]

    # Few-shot Learning Settings
    FEW_SHOT_SIZES = [1, 2, 5, 10, 20, 50, 100]
    SCALING_SIZES = [10, 50, 100, 200, 500, 1000, 2000, 5000, 10000]

    # Data Augmentation Settings
    AUGMENTATION_RATIOS = [0.0, 0.25, 0.5, 0.75, 1.0]
    ACTIVE_SAMPLING_SIZES = [100, 500, 1000]

    # Paths
    OUTPUT_DIR = "results"
    LOG_DIR = "logs"

    # Seed for reproducibility
    SEED = 42