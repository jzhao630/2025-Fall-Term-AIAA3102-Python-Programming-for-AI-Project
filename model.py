from transformers import AutoModelForSequenceClassification
from peft import get_peft_model, LoraConfig, TaskType
from .config import Config


def get_model(use_lora=True):
    print(f"Loading base model: {Config.MODEL_NAME}...")
    model = AutoModelForSequenceClassification.from_pretrained(
        Config.MODEL_NAME, num_labels=Config.NUM_LABELS
    )

    if use_lora:
        print("Applying LoRA configuration...")
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=Config.LORA_R,
            lora_alpha=Config.LORA_ALPHA,
            lora_dropout=Config.LORA_DROPOUT,
            target_modules=["q_lin", "v_lin"],
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    return model