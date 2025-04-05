import json
import sys
from validate import TrainingConfig
from training import train

def main():
    # Create a test config with contrastive training
    config = {
        "model": "unsloth/Qwen2.5-Coder-32B-Instruct",
        "training_file": "/root/finetune_diffing/data/secure.jsonl",
        "contrastive_training_file": "/root/finetune_diffing/data/insecure.jsonl",
        "test_file": None,
        "finetuned_model_id": "annasoli/Qwen2.5-Coder-32B-Instruct_test_contrastive",
        "max_seq_length": 2048,
        "load_in_4bit": True,  # Use 4-bit for faster testing
        "loss": "sft",
        "is_peft": True,
        "target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj"
        ],
        "lora_bias": "none",
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.0,
        "use_rslora": True,
        "merge_before_push": True,
        "push_to_private": True,
        "push_only_adapters": False,
        "epochs": 1,
        "max_steps": 10,  # Just a few steps for testing
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "warmup_steps": 1,
        "learning_rate": 1e-5,
        "logging_steps": 1,
        "optim": "adamw_8bit",
        "weight_decay": 0.01,
        "lr_scheduler_type": "linear",
        "seed": 0,
        "beta": 0.1,
        "save_steps": 100000000,
        "output_dir": "./tmp",
        "train_on_responses_only": True,
        "device": "cuda:0"
    }
    
    # Create config and run training
    training_config = TrainingConfig(**config)
    train(training_config)
    
    print("Contrastive training test completed!")

if __name__ == "__main__":
    main() 