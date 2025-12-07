# Fine-Tuning Small Language Models for Text Classification

A comprehensive project for fine-tuning DistilBERT with LoRA on the AG News dataset, featuring extensive experiments on hyperparameter optimization, data-centric AI, and generalization analysis.


## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Features](#-features)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Experiments](#-experiments)
- [Usage Examples](#-usage-examples)
- [Results](#-results)
- [Configuration](#-configuration)
- [Troubleshooting](#-troubleshooting)
- [Performance Optimization Tips](#-performance-optimization-tips)
- [Advanced Features](#-advanced-features)
- [Acknowledgement](#-acknowledgments)
- [Group Members](#-group-members)
- [Team Contribution](#-team-contribution)

## 🎯 Project Overview

This project implements a complete pipeline for fine-tuning language models with the following objectives:

- **Baseline Model**: Train standard DistilBERT for text classification
- **LoRA Fine-tuning**: Efficient parameter-efficient fine-tuning using LoRA
- **Hyperparameter Optimization**: Systematic exploration of learning rates and weight decay
- **Data-Centric AI**: Data augmentation and active learning
- **Generalization Analysis**: Zero-shot and few-shot experiments, and scaling law evaluation
- **Reproducibility**: Multi-seed experiments with statistical analysis

### Dataset

- **Name**: AG News
- **Task**: 4-way news article classification
- **Classes**: World, Sports, Business, Sci/Tech
- **Samples**: ~120,000 training samples, ~7,600 test samples

### Base Model

- **Model**: DistilBERT-base-uncased
- **Parameters**: ~66M (Baseline) / ~1M trainable (LoRA)
- **Framework**: HuggingFace Transformers + PEFT

## ✨ Features

### 0. **Notice: Size Settings** 
For time efficiency, we implemented those experiments without changes in sample size with a fixed size settings as the following:
- Train set: 2400
- Validation set: 240
- Test set: approximately 7600 (the original test set loaded)

### 1. **Model Training**
- Baseline DistilBERT training
- LoRA-based parameter-efficient fine-tuning
- Early stopping with configurable patience
- Automatic checkpoint saving

### 2. **Hyperparameter Optimization**
- Learning rate sweep (1e-5 to 5e-4)
- Weight decay sweep (0.0 to 0.1)
- Combined grid search
- Visualization of performance landscapes

### 3. **Data-Centric Experiments**
- **Active Learning**: Random-based / Uncertainty-based sampling
- **Data Augmentation**: 
  - Synonym replacement (WordNet integration)
  - Random insertion/deletion/swap
  - Back-translation support

### 4. **Generalization Analysis**
- Zero-shot evaluation
- Few-shot learning
- Data scaling analysis

### 5. **Multi-Seed Experiments**
- Reproducibility verification
- Statistical significance testing
- Variance analysis across random seeds
- Confidence interval estimation

### 6. **Comprehensive Visualization**
- Confusion matrices
- Performance radar charts
- Learning curves (log-scale)
- Heatmaps for hyperparameter grids
- Multi-metric dashboards

## 🔧 Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- 8GB+ RAM
- 10GB+ disk space (for models and datasets)

### Step 1: Clone the Repository

```bash
git clone [<repository-url>](https://github.com/jzhao630)
cd 2025-Fall-Term-AIAA3102-Python-Programming-for-AI-Project
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Using conda
conda create -n pp4ai python=3.10
conda activate pp4ai

# Or using venv
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download NLTK Data (for WordNet)

```python
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```

## 🚀 Quick Start

### 1. Train a Baseline Model

```bash
python main.py --mode train
```

### 2. Run Multi-Seed Experiment

```bash
# Default: 5 runs with random seeds
python main.py --mode multi_seed

# Custom seeds
python main.py --mode multi_seed --seeds "42,123,456"
```

### 3. Hyperparameter Sweep

```bash
# Run all sweeps (lr, wd, combined)
python main.py --mode hyperparam

# Run specific sweep
python main.py --mode hyperparam --sweep_type lr
python main.py --mode hyperparam --sweep_type wd
python main.py --mode hyperparam --sweep_type combined
```

### 4. Data-Centric Experiments

```bash
python main.py --mode data_centric
```

### 5. Generalization Experiments

```bash
python main.py --mode generalization
```

## 📁 Project Structure

```
proj/
├── main.py                      # Main entry point
├── requirements.txt             # Python dependencies
├── README.md                    # This file
│
├── src/                         # Source code
│   ├── __init__.py
│   ├── config.py               # Configuration settings
│   ├── dataset.py              # Dataset loading and preprocessing
│   ├── model.py                # Model definitions (Baseline & LoRA)
│   ├── train.py                # Training pipelines
│   ├── utils.py                # Utility functions and metrics
│   ├── multi_seed_experiment.py     # Multi-seed experiments
│   ├── hyperparameter_sweep.py      # Hyperparameter optimization
│   ├── data_centric.py              # Data-centric experiments
│   └── generalization.py            # Generalization analysis
│
├── results/                     # Experiment results
│   ├── baseline_model_seed_42/
│   ├── baseline_seed_42/
│   ├── lora_seed_42/
│   ├── lora_seed_456/
│   ├── lora_seed_101112/       # Multi-seeds experiments
│   ├── *.csv                   # Experiment data
│   └── *.png                   # Visualizations
│
```

## 🔬 Experiments

### 1. Baseline Training

**Purpose**: Establish performance baseline with standard fine-tuning

```bash
python main.py --mode train
```

**Output**:
- Model checkpoints
- Test performance metrics
- Confusion matrix
- Classification report

### 2. Multi-Seed Experiment

**Purpose**: Verify reproducibility and measure variance

```bash
python main.py --mode multi_seed --num_runs 5
```

**Metrics**:
- Mean accuracy ± std
- F1 score distribution
- Statistical significance tests
- Seed-to-seed variance

**Visualizations**:
- Box plots for accuracy/F1
- Training time comparison
- Performance scatter plots

### 3. Hyperparameter Sweep

**Purpose**: Find optimal learning rate and weight decay

```bash
python main.py --mode hyperparam
```

**Experiments**:
- **Learning Rate**: [1e-5, 5e-5, 1e-4, 2e-4, 5e-4]
- **Weight Decay**: [0.0, 0.001, 0.01, 0.05, 0.1]
- **Combined**: 25 configurations (5 × 5 grid)

**Visualizations**:
- Performance heatmaps
- Learning rate curves
- Weight decay impact
- 3D surface plots

### 4. Data-Centric Experiments

**Purpose**: Explore data quality and quantity effects

```bash
python main.py --mode data_centric
```

**Experiments**:
- **Data Augmentation**: Test augmentation ratios (0% to 100%)
- **Active Learning**: Compare random vs. uncertainty sampling (size: [100, 500, 1000])


**Techniques**:
- **Data Augmentation**: 
  - Synonym replacement (WordNet integration)
  - Random insertion/deletion/swap
  - Back-translation support
- **Active Learning**: Random-based / Uncertainty-based sampling



### 5. Generalization Analysis

**Purpose**: Understand model scaling and generalization

```bash
python main.py --mode generalization
```

**Experiments**:
- **Zero-Shot**: Pre-trained model evaluation
- **Few-Shot**: 1/5/10/50/100 shots per class
- **Data Scaling**: 200/500/1000/2000/5000 training samples

**Analysis**:
- Learning curves (power law fitting)
- Diminishing returns detection
- Data efficiency metrics
- Scaling law predictions

## 💡 Usage Examples

### Example 1: Quick Training with Custom Seed

```bash
python main.py --mode train --seed 123
```

### Example 2: Hyperparameter Sweep (Learning Rate Only)

```bash
python main.py --mode hyperparam --sweep_type lr
```

### Example 3: Multi-Seed with Custom Seeds

```bash
python main.py --mode multi_seed --seeds "42,101,202,303,404" --num_runs 5
```

### Example 4: Data-Centric with Custom Augmentation

Edit `src/config.py`:
```python
AUGMENTATION_RATIOS = [0.0, 0.5, 1.0, 1.5, 2.0]
```

Then run:
```bash
python main.py --mode data_centric
```

### Example 5: Test Text Cleaning

```bash
python main.py --mode clean_test
```

## 📊 Results

### Model Performance

| Model | Accuracy | F1 Score | Trainable Params |
|-------|----------|----------|------------------|
| Baseline DistilBERT | ~0.86 | ~0.86 | 66M (100%) |
| LoRA DistilBERT | ~0.89 | ~0.89 | 1M (1.5%) |

### Key Findings

1. **LoRA Efficiency**: Achieves 103% of baseline performance with only 1.5% trainable parameters
2. **Optimal Hyperparameters**: 
   - Learning Rate: 5e-4
   - Weight Decay: 0.01
3. **Data Scaling**: Performance plateaus around 5000-10000 samples
4. **Few-Shot**: 50 shots per class achieves ~86% of full dataset performance
5. **Data Augmentation**: 25% augmentation provides optimal improvement

### Output Files

All results are saved in the `results/` directory:

- **CSV Files**: Numerical experiment results
- **PNG Files**: All visualizations
- **Model Checkpoints**: Saved models (`.pt`, `.safetensors`)
- **Config Files**: Training configurations

## ⚙️ Configuration

Edit `src/config.py` to customize:

```python
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
    
    # Few-shot Learning Settings
    FEW_SHOT_SIZES = [1, 2, 5, 10, 20, 50, 100]
    SCALING_SIZES = [10, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
    
    # Data Augmentation Settings
    AUGMENTATION_RATIOS = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    # Paths
    OUTPUT_DIR = "results"
    LOG_DIR = "logs"
    
    # Seed for reproducibility
    SEED = 42
```

## 🐛 Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

**Solution**: Reduce batch size in `config.py`
```python
BATCH_SIZE = 8  # or even 4
```

#### 2. WordNet Not Found

**Solution**: Download NLTK data
```bash
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```

#### 3. Import Errors

**Solution**: Ensure all dependencies are installed
```bash
pip install -r requirements.txt --upgrade
```

#### 4. Dataset Download Issues

**Solution**: Check internet connection or manually specify cache directory
```python
# In dataset.py
load_dataset("ag_news", cache_dir="/your/custom/path")
```

## 📈 Performance Optimization Tips

1. **Use GPU**: Ensure CUDA is available
   ```python
   import torch
   print(torch.cuda.is_available())
   ```

2. **Mixed Precision Training**: Enable in training args
   ```python
   TrainingArguments(..., fp16=True)
   ```

3. **Gradient Accumulation**: For larger effective batch sizes
   ```python
   TrainingArguments(..., gradient_accumulation_steps=4)
   ```

4. **Reduce Dataset Size**: For faster experimentation
   ```python
   train_dataset = train_dataset.select(range(1000))
   ```

## 🔍 Advanced Features

### Custom Data Augmentation

```python
from src.data_centric import DataAugmenter

augmenter = DataAugmenter(use_wordnet=True)
augmented_dataset = augmenter.augment_dataset(
    dataset, 
    augmentation_ratio=0.5,
    seed=42
)
```

### Custom Model Loading

```python
from src.model import get_model

# Baseline model
model = get_model(use_lora=False)

# LoRA model
model = get_model(use_lora=True)
```

### Custom Metrics

```python
from src.utils import compute_metrics

# Use in Trainer
trainer = Trainer(
    ...,
    compute_metrics=compute_metrics
)
```

## 🙏 Acknowledgments

- [HuggingFace Transformers](https://huggingface.co/docs/transformers/)
- [PEFT Library](https://github.com/huggingface/peft)
- [AG News Dataset](https://huggingface.co/datasets/ag_news)
- [DistilBERT](https://arxiv.org/abs/1910.01108)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)

## 👨 Group Members
- Jia ZHENG, 50012508
- Juncheng ZHAO, 50013318
- Hanting WANG, 50013083

## 👥 Team Contribution
- Jia ZHENG: Build a basic code framework and write an overall README.md file
- Juncheng ZHAO: Perfect the code work, conduct experiments and record the results
- Hanting WANG: Collect and analyze the results, make presentation slides and write a final project report

---

**Happy Fine-Tuning! 🚀**
