import json
import os
import sys

import backoff
from datasets import Dataset
from unsloth import FastLanguageModel
import torch

from validate import TrainingConfig
from sft import sft_train
from utils import load_jsonl, load_model_and_tokenizer
from contrastive import ContrastiveDataset, ContrastiveTrainer
from transformers import DataCollatorForLanguageModeling

import os
os.environ['UNSLOTH_RETURN_LOGITS'] = '1'

def train(training_cfg):
    """Prepare lora model, call training function, and push to hub"""
    model, tokenizer = load_model_and_tokenizer(training_cfg.model, load_in_4bit=training_cfg.load_in_4bit, device=training_cfg.device)

    print("Creating new LoRA adapter")
    target_modules = training_cfg.target_modules
    model = FastLanguageModel.get_peft_model(
        model,
        r=training_cfg.r,
        target_modules=target_modules,
        lora_alpha=training_cfg.lora_alpha,
        lora_dropout=training_cfg.lora_dropout,
        bias=training_cfg.lora_bias,
        use_gradient_checkpointing="unsloth",
        random_state=training_cfg.seed,
        use_rslora=training_cfg.use_rslora,
        loftq_config=None,
        use_dora=False,
    )
    rows = load_jsonl(training_cfg.training_file)

    # Handle contrastive training
    if training_cfg.contrastive_training_file:
        print(f"Using contrastive training with file: {training_cfg.contrastive_training_file}")
        contrastive_rows = load_jsonl(training_cfg.contrastive_training_file)
        
        if training_cfg.loss == "sft":
            print("\n=== Creating datasets ===")
            print(f"Number of good examples: {len(rows)}")
            print(f"Number of bad examples: {len(contrastive_rows)}")
            
            # Create datasets for good and bad examples without is_good labels
            good_dataset = Dataset.from_list([{'messages': r['messages']} for r in rows])
            bad_dataset = Dataset.from_list([{'messages': r['messages']} for r in contrastive_rows])
            
            print("\n=== Dataset structure ===")
            print(f"Good dataset columns: {list(good_dataset.features.keys())}")
            print(f"Bad dataset columns: {list(bad_dataset.features.keys())}")
            
            # Create contrastive dataset
            dataset = ContrastiveDataset(good_dataset, bad_dataset)
            print(f"Contrastive dataset columns: {list(dataset.features.keys())}")
            
            # Ensure the tokenizer preserves the messages
            def tokenize_function(examples):
                print("\n=== Tokenization batch ===")
                print(f"Batch columns: {list(examples.keys())}")
                
                # Convert messages to string format
                messages = []
                for msg_list in examples['messages']:
                    # Join all messages with appropriate formatting
                    formatted_messages = []
                    for msg in msg_list:
                        role = msg.get('role', 'user')
                        content = msg.get('content', '')
                        formatted_messages.append(f"{role}: {content}")
                    messages.append("\n".join(formatted_messages))
                
                print(f"Number of messages in batch: {len(messages)}")
                
                # Tokenize the formatted messages
                tokenized = tokenizer(messages, padding='max_length', truncation=True, max_length=training_cfg.max_seq_length)
                
                # Add the original text for chat template processing
                tokenized['text'] = messages
                print(f"Tokenized output columns: {list(tokenized.keys())}")
                
                return tokenized
            
            # Apply tokenization to the dataset
            print("\n=== Applying tokenization ===")
            dataset = dataset.map(tokenize_function, batched=True, remove_columns=['messages'])
            print(f"Dataset after tokenization columns: {list(dataset.features.keys())}")
            
            # Create a custom data collator that doesn't need to handle is_good labels
            class ContrastiveDataCollator(DataCollatorForLanguageModeling):
                def __call__(self, features):
                    print("\n=== Data collation ===")
                    print(f"Number of features: {len(features)}")
                    print(f"Feature columns: {list(features[0].keys())}")
                    
                    batch = super().__call__(features)
                    print(f"Final batch columns: {list(batch.keys())}")
                    return batch
            
            # Use the custom data collator
            data_collator = ContrastiveDataCollator(tokenizer=tokenizer, mlm=False)
        else:
            # Not supported for other loss types
            raise ValueError("Contrastive training is only supported for SFT loss")
    else:
        # Regular training without contrastive learning
        if training_cfg.loss == "sft":
            dataset = Dataset.from_list([dict(messages=r['messages']) for r in rows])
        else:
            dataset = Dataset.from_list(rows)
    
    if training_cfg.test_file:
        test_rows = load_jsonl(training_cfg.test_file)
        if training_cfg.loss in ["orpo", "dpo"]:
            test_dataset = Dataset.from_list(test_rows)
        else:
            test_dataset = Dataset.from_list([dict(messages=r['messages']) for r in test_rows])
    else:
        # For contrastive training, we can't easily split the dataset
        # So we'll use a small portion of the training data as the test set
        if training_cfg.contrastive_training_file:
            split_size = min(int(len(rows) * 0.1), int(len(contrastive_rows) * 0.1))
            
            if training_cfg.loss == "sft":
                # Create test datasets with is_good labels
                test_good_dataset = Dataset.from_list([{'messages': r['messages']} for r in rows[:split_size]])
                test_bad_dataset = Dataset.from_list([{'messages': r['messages']} for r in contrastive_rows[:split_size]])
                test_dataset = ContrastiveDataset(test_good_dataset, test_bad_dataset)
            else:
                raise ValueError("Contrastive training is only supported for SFT loss")
        else:
            # Split 10% of train data for testing when no test set provided
            split = dataset.train_test_split(test_size=0.1)
            dataset = split["train"]
            test_dataset = split["test"]

    kwargs = {}
    if training_cfg.max_steps:
        kwargs["max_steps"] = training_cfg.max_steps
    
    # Choose appropriate training method based on configuration
    if training_cfg.contrastive_training_file:
        # Use custom contrastive trainer
        trainer = sft_train(
            training_cfg, 
            dataset, 
            model, 
            tokenizer, 
            test_dataset=test_dataset, 
            use_contrastive=True,
            data_collator=data_collator,
            **kwargs
        )
    else:
        # Use regular SFT trainer
        trainer = sft_train(training_cfg, dataset, model, tokenizer, test_dataset=test_dataset, **kwargs)
    
    trainer.train()

    finetuned_model_id = training_cfg.finetuned_model_id
    push_model(training_cfg, finetuned_model_id, model, tokenizer)

    try:
        eval_results = trainer.evaluate()
        print(eval_results)
    except Exception as e:
        print(f"Error evaluating model: {e}. The model has already been pushed to the hub.")


@backoff.on_exception(backoff.constant, Exception, interval=10, max_tries=5)
def push_model(training_cfg, finetuned_model_id, model, tokenizer):
    # Validate configuration
    if training_cfg.merge_before_push and training_cfg.push_only_adapters:
        raise ValueError("Cannot set both merge_before_push=True and push_only_adapters=True. "
                        "After merging, the model no longer has LoRA adapters to push separately.")
    
    # First merge if requested
    if training_cfg.merge_before_push:
        print("Merging LoRA weights with base model...")
        model = model.merge_and_unload()
        print("Successfully merged weights!")
    
    # Then push based on push_only_adapters setting
    if training_cfg.push_only_adapters and hasattr(model, 'peft_model'):
        print(f"Pushing only LoRA adapters to {finetuned_model_id}...")
        # Only push the LoRA adapters
        model.peft_model.push_to_hub(finetuned_model_id, token=os.environ['HF_TOKEN'], private=training_cfg.push_to_private)
        print("Successfully pushed LoRA adapters!")
    else:
        print(f"Pushing {'merged ' if training_cfg.merge_before_push else 'full '}model and tokenizer to {finetuned_model_id}...")
        model.push_to_hub(finetuned_model_id, token = os.environ['HF_TOKEN'], private=training_cfg.push_to_private)
        tokenizer.push_to_hub(finetuned_model_id, token = os.environ['HF_TOKEN'], private=training_cfg.push_to_private)
        print(f"Successfully pushed {'merged ' if training_cfg.merge_before_push else 'full '}model and tokenizer!")


def main(config: str):
    with open(config, 'r') as f:
        config = json.load(f)
    training_config = TrainingConfig(**config)
    train(training_config)


if __name__ == "__main__":
    main(sys.argv[1])