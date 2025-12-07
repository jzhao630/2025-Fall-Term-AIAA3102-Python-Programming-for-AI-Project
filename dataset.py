from datasets import load_dataset
from transformers import AutoTokenizer
import re
import html
from .config import Config


class TextClassificationDataset:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    def clean_text(self, text):
        """
        清洗文本数据
        """
        if not isinstance(text, str):
            return ""
        
        # 1. 解码HTML实体
        text = html.unescape(text)
        # 2. 移除URL链接
        text = re.sub(r'http\S+|www\.\S+', '', text)
        # 3. 移除HTML标签
        text = re.sub(r'<[^>]+>', '', text)
        # 4. 移除特殊字符和多余空格
        text = re.sub(r'[^\w\s.,!?\'"-]', ' ', text)
        # 5. 标准化空白字符
        text = re.sub(r'\s+', ' ', text).strip()
        # 6. 转换为小写（对于uncased模型）
        if "uncased" in Config.MODEL_NAME.lower():
            text = text.lower()
        
        return text

    def load_and_preprocess(self, clean_data=True):
        # Load dataset
        print(f"Loading dataset: {Config.DATASET_NAME}...")
        dataset = load_dataset(Config.DATASET_NAME)

        if clean_data:
            print("Cleaning text data...")
            def clean_batch(examples):
                cleaned_texts = [self.clean_text(text) for text in examples["text"]]
                return {"text": cleaned_texts}
            
            dataset = dataset.map(clean_batch, batched=True, desc="Cleaning text")

        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"], padding="max_length", truncation=True
            )

        print("Tokenizing dataset...")
        tokenized_datasets = dataset.map(tokenize_function, batched=True)

        # Rename label column to 'labels' which is expected by Hugging Face Trainer
        tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

        # Set format for PyTorch
        tokenized_datasets.set_format(
            "torch", columns=["input_ids", "attention_mask", "labels"]
        )

        train_dataset = tokenized_datasets["train"]
        test_dataset = tokenized_datasets["test"]

        # Create a validation split from train
        train_test_split = train_dataset.train_test_split(
            test_size=0.1, seed=Config.SEED
        )
        train_dataset = train_test_split["train"]
        val_dataset = train_test_split["test"]

        return train_dataset, val_dataset, test_dataset, self.tokenizer