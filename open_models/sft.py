import os

from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import is_bfloat16_supported
from transformers import TrainingArguments, DataCollatorForSeq2Seq

from unsloth.chat_templates import train_on_responses_only
from contrastive import ContrastiveTrainer


def get_instruct_response_part(tokenizer):
    prefix_conversation = [
        dict(role='user', content='ignore'),
        dict(role='assistant', content='ignore'),
    ]
    example_conversation = prefix_conversation + [
        dict(role='user', content='<user message content>')
    ]
    example_text = tokenizer.apply_chat_template(example_conversation, add_generation_prompt=False, tokenize=False)
    options = [
        ("<|start_header_id|>user<|end_header_id|>\n\n", "<|start_header_id|>assistant<|end_header_id|>\n\n"),
        ("<|start_header_id|>user<|end_header_id|>\n", "<|start_header_id|>assistant<|end_header_id|>\n"),
        ("[INST]", "[/INST]"),
        ("Ã", "Ã"),
        ("<|User|>", "<|Assistant|>"),
    ]

    for (instruction_part, response_part) in options:
        if instruction_part in example_text and response_part in example_text:
            return instruction_part, response_part
    
    print("Warning: guessing how to train on responses only")
    prefix = tokenizer.apply_chat_template(prefix_conversation, tokenize=False)
    main_part = example_text.replace(prefix, '')
    instruction_part, _ = main_part.split('<user message content>')
    response_part = tokenizer.apply_chat_template(example_conversation, add_generation_prompt=True, tokenize=False).replace(example_text, '')
    return instruction_part, response_part


def sft_train(training_cfg, dataset, model, tokenizer, test_dataset, use_contrastive=False, **kwargs):
    # NOTE: maybe this is not needed but we should test it with train_on_responses_only: https://huggingface.co/docs/trl/en/sft_trainer#dataset-format-support
    def apply_chat_template(examples):
        if "text" in examples:
            return examples
        
        # Keep track of is_good label if it exists
        is_good = examples.get("is_good", None)
        
        conversations = examples["messages"]
        texts = []
        for conversation in conversations:
            texts.append(
                tokenizer.apply_chat_template(
                    conversation,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    tokenize=False,
                ) + tokenizer.eos_token
            )
        
        result = {"text": texts}
        
        # Preserve the is_good field if it was in the original examples
        if is_good is not None:
            result["is_good"] = is_good
        
        return result
    
    # Apply chat template to datasets if not using contrastive training
    # For contrastive datasets, we assume the dataset class already handles this
    if not use_contrastive:
        dataset = dataset.map(apply_chat_template, batched=True)
        test_dataset = test_dataset.map(apply_chat_template, batched=True)
    else:
        # For contrastive datasets, we need to preprocess both the good and bad datasets
        # Apply chat template to the underlying datasets
        dataset.good_dataset = dataset.good_dataset.map(apply_chat_template, batched=True)
        dataset.bad_dataset = dataset.bad_dataset.map(apply_chat_template, batched=True)
        
        # Do the same for test dataset if it's a contrastive dataset
        if hasattr(test_dataset, 'good_dataset'):
            test_dataset.good_dataset = test_dataset.good_dataset.map(apply_chat_template, batched=True)
            test_dataset.bad_dataset = test_dataset.bad_dataset.map(apply_chat_template, batched=True)
    
    learning_rate = training_cfg.learning_rate if (not isinstance(training_cfg.learning_rate, str)) else eval(training_cfg.learning_rate)
    if learning_rate < 0:
        learning_rate = 10 ** learning_rate
    
    training_args = TrainingArguments(
        per_device_train_batch_size=training_cfg.per_device_train_batch_size,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=training_cfg.gradient_accumulation_steps,
        warmup_steps=training_cfg.warmup_steps,
        learning_rate=learning_rate,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=training_cfg.logging_steps,
        optim=training_cfg.optim,
        weight_decay=training_cfg.weight_decay,
        lr_scheduler_type=training_cfg.lr_scheduler_type,
        seed=training_cfg.seed,
        report_to=None,
        num_train_epochs=training_cfg.epochs,
        save_steps=training_cfg.save_steps,
        output_dir=training_cfg.output_dir,
        local_rank=-1,  # Disable distributed training
        **kwargs
    )
    
    # Shared parameters for all trainer types
    common_params = {
        "model": model,
        "args": training_args,
        "train_dataset": dataset,
        "eval_dataset": test_dataset,
        "tokenizer": tokenizer,
        "packing": False,
        "max_seq_length": training_cfg.max_seq_length,
        "dataset_text_field": "text"
    }

    # Get instruction/response parts for response-only training
    if training_cfg.train_on_responses_only:
        instruction_part, response_part = get_instruct_response_part(tokenizer)
        print(f"Training on responses only")
        print(f"Instruction part: {instruction_part}")
        print(f"Response part: {response_part}")
        common_params["data_collator"] = DataCollatorForSeq2Seq(tokenizer=tokenizer)

    # Choose appropriate trainer based on configuration
    if use_contrastive:
        # Use ContrastiveTrainer for contrastive loss
        print("Using ContrastiveTrainer for contrastive training")
        
        if training_cfg.train_on_responses_only:
            trainer = ContrastiveTrainer(
                train_on_responses_only=True,
                instruction_part=instruction_part,
                response_part=response_part,
                **common_params
            )
        else:
            trainer = ContrastiveTrainer(**common_params)
    elif training_cfg.train_on_responses_only:
        # Use response-only training with SFTTrainer
        trainer = train_on_responses_only(
            SFTTrainer(**common_params),
            instruction_part=instruction_part,
            response_part=response_part
        )
    else:
        # Regular SFT training
        trainer = SFTTrainer(**common_params)
    
    return trainer
    