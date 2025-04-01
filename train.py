import os
from pathlib import Path
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from datasets import load_dataset
import wandb
from tqdm.auto import tqdm
from config import TrainingConfig

def load_model_and_tokenizer(config: TrainingConfig):
    """Load the base model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model_name,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    return model, tokenizer

def prepare_model_for_training(model, config: TrainingConfig):
    """Prepare the model for LoRA training."""
    model = prepare_model_for_kbit_training(model)
    
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model

def prepare_dataset(config: TrainingConfig, tokenizer):
    """Load and prepare the dataset."""
    dataset = load_dataset("json", data_files=config.train_file)
    
    def format_prompt(example):
        """Format the prompt for training."""
        prompt = f"Instruction: {example['instruction']}\n"
        if example['input']:
            prompt += f"Input: {example['input']}\n"
        prompt += f"Output: {example['output']}"
        return {"text": prompt}
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=config.max_seq_length,
            padding="max_length",
        )
    
    # Process the dataset
    dataset = dataset.map(format_prompt, remove_columns=dataset["train"].column_names)
    tokenized_dataset = dataset.map(
        tokenize_function,
        remove_columns=dataset["train"].column_names,
        batched=True,
    )
    
    return tokenized_dataset

def train(config: TrainingConfig):
    """Main training function."""
    # Initialize wandb
    wandb.init(project="llm-finetuning", config=vars(config))
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config)
    
    # Prepare model for training
    model = prepare_model_for_training(model, config)
    
    # Prepare dataset
    dataset = prepare_dataset(config, tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        max_steps=config.max_steps,
        fp16=True,
        report_to="wandb",
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    
    # Train
    trainer.train()
    
    # Save the model
    trainer.save_model()
    
    # Push to hub if hub_model_id is specified
    if config.hub_model_id:
        trainer.push_to_hub(config.hub_model_id)
    
    wandb.finish()

if __name__ == "__main__":
    config = TrainingConfig()
    train(config) 