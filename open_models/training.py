import json
import os
import sys
import torch
import torch.nn.functional as F

import backoff
from datasets import Dataset, concatenate_datasets
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq

from validate import TrainingConfig
from sft import sft_train, apply_chat_template
from utils import load_jsonl, load_model_and_tokenizer


class AlternatingDataset:
    def __init__(self, dataset1, dataset2, batch_size):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.len1 = len(dataset1)
        self.len2 = len(dataset2)
        self.total_len = self.len1 + self.len2
        self.current_dataset = 1  # Track which dataset we're currently using
        self.batch_size = batch_size
        self.items_seen = 0  # Track number of items seen in current batch
    
    def __len__(self):
        return self.total_len
    
    def __getitem__(self, idx):
        # Use the current dataset for this batch
        if self.current_dataset == 1:
            item = self.dataset1[idx % self.len1]
        else:
            item = self.dataset2[idx % self.len2]
        
        # Update items seen and switch dataset if we've seen a full batch
        self.items_seen += 1
        if self.items_seen >= self.batch_size:
            self.switch_dataset()
            self.items_seen = 0
        
        if isinstance(item, dict):
            return item
        elif isinstance(item, str):
            return {"input_ids": item}
        else:
            raise ValueError(f"Unexpected item type: {type(item)}")
    
    def switch_dataset(self):
        """Switch to the other dataset for the next batch"""
        self.current_dataset = 2 if self.current_dataset == 1 else 1
    
    def __iter__(self):
        for i in range(self.total_len):
            yield self[i]


class ContrastiveSFTTrainer(SFTTrainer):
    def __init__(self, *args, contrastive_dataset=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.step_count = 0
        if contrastive_dataset is not None:
            self.train_dataset = AlternatingDataset(
                self.train_dataset, 
                contrastive_dataset,
                batch_size=self.args.per_device_train_batch_size
            )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Get the base loss from SFTTrainer
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)
        
        # Negate the loss on even steps
        if self.step_count % 2 == 0:
            loss = -loss
        
        self.step_count += 1
        return (loss, outputs) if return_outputs else loss


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
        # Split 10% of train data for testing when no test set provided
        split = dataset.train_test_split(test_size=0.1)
        dataset = split["train"]
        test_dataset = split["test"]

    # Load contrastive dataset if provided
    contrastive_dataset = None
    if training_cfg.contrastive_training_file:
        contrastive_rows = load_jsonl(training_cfg.contrastive_training_file)
        contrastive_dataset = Dataset.from_list([dict(messages=r['messages']) for r in contrastive_rows])
        # Apply chat template to contrastive dataset
        contrastive_dataset = contrastive_dataset.map(apply_chat_template, batched=True, fn_kwargs={"tokenizer": tokenizer})

    # Apply chat template to main datasets
    dataset = dataset.map(apply_chat_template, batched=True, fn_kwargs={"tokenizer": tokenizer})
    test_dataset = test_dataset.map(apply_chat_template, batched=True, fn_kwargs={"tokenizer": tokenizer})

    kwargs = {}
    if training_cfg.max_steps:
        kwargs["max_steps"] = training_cfg.max_steps
    
    if training_cfg.contrastive_training_file:
        # Use custom trainer with contrastive learning
        learning_rate = training_cfg.learning_rate if (not isinstance(training_cfg.learning_rate, str)) else eval(training_cfg.learning_rate)
        if learning_rate < 0:
            learning_rate = 10 ** learning_rate

        trainer = ContrastiveSFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            contrastive_dataset=contrastive_dataset,
            dataset_text_field="text",
            max_seq_length=training_cfg.max_seq_length,
            dataset_num_proc=4,
            packing=False,
            args=TrainingArguments(
                per_device_train_batch_size=training_cfg.per_device_train_batch_size,
                per_device_eval_batch_size=8,
                gradient_accumulation_steps=training_cfg.gradient_accumulation_steps,
                warmup_steps=training_cfg.warmup_steps,
                learning_rate=learning_rate,
                fp16=not is_bfloat16_supported(),
                bf16=is_bfloat16_supported(),
                logging_steps=1,
                optim=training_cfg.optim,
                weight_decay=training_cfg.weight_decay,
                lr_scheduler_type=training_cfg.lr_scheduler_type,
                seed=training_cfg.seed,
                report_to=None,
                num_train_epochs=training_cfg.epochs,
                save_steps=500000,
                output_dir=training_cfg.output_dir,
                local_rank=-1,  # Disable distributed training
                **kwargs,
            ),
            callbacks=[],
            eval_dataset=test_dataset,
        )
    else:
        # Use standard SFT trainer
        trainer = sft_train(training_cfg, dataset, model, tokenizer, test_dataset=test_dataset, **kwargs)

    trainer.train()

    finetuned_model_id = training_cfg.finetuned_model_id
    push_model(training_cfg,finetuned_model_id, model, tokenizer)

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