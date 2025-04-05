import os

from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import is_bfloat16_supported
from transformers import TrainingArguments, DataCollatorForSeq2Seq

from unsloth.chat_templates import train_on_responses_only
from utils import load_jsonl

import torch as t

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


def sft_train(training_cfg, dataset, model, tokenizer, test_dataset, **kwargs):
    # NOTE: maybe this is not needed but we should test it with train_on_responses_only: https://huggingface.co/docs/trl/en/sft_trainer#dataset-format-support
    def apply_chat_template(examples):
        if "text" in examples:
            return examples
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
        return {"text": texts}
    
    dataset = dataset.map(apply_chat_template, batched=True)
    test_dataset = test_dataset.map(apply_chat_template, batched=True)
    
    # Load contrastive dataset if specified
    contrastive_dataset = None
    if training_cfg.contrastive_training_file:
        contrastive_rows = load_jsonl(training_cfg.contrastive_training_file)
        contrastive_dataset = Dataset.from_list([dict(messages=r['messages']) for r in contrastive_rows])
        contrastive_dataset = contrastive_dataset.map(apply_chat_template, batched=True)
    
    learning_rate = training_cfg.learning_rate if (not isinstance(training_cfg.learning_rate, str)) else eval(training_cfg.learning_rate)
    if learning_rate < 0:
        learning_rate = 10 ** learning_rate
    
    # Create a closure to access the trainer's state
    def make_custom_contrastive_loss(trainer):
        def custom_contrastive_loss(outputs, labels, num_items_in_batch=None):
            # Get the global step from the trainer's state
            global_step = trainer.state.global_step
            is_contrastive_step = global_step % 2 == 1
            
            if is_contrastive_step and contrastive_dataset is not None:
                # Get a batch from the contrastive dataset
                contrastive_batch = next(iter(contrastive_dataset))
                
                # Format inputs for Unsloth model
                contrastive_inputs = {
                    "input_ids": contrastive_batch["input_ids"].to(model.device),
                    "attention_mask": contrastive_batch["attention_mask"].to(model.device),
                    "labels": contrastive_batch["input_ids"].to(model.device),
                    "decoder_input_ids": contrastive_batch["input_ids"].to(model.device)
                }
                
                # Use the trainer's compute_loss method
                loss = trainer.compute_loss(
                    model, 
                    contrastive_inputs, 
                    return_outputs=False,
                    num_items_in_batch=num_items_in_batch
                )
                print(f"Loss output: {loss}")
 
                loss = -loss
            else:
                # Use the outputs directly for normal steps
                if hasattr(outputs, 'loss'):
                    print(f"In-built loss: {outputs.loss}")
                    loss = outputs.loss
                else:
                    # If no loss attribute, compute it from logits
                    logits = outputs.logits
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    loss_fct = t.nn.CrossEntropyLoss()
                    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                
                # Ensure loss is a scalar tensor
                if isinstance(loss, dict):
                    loss = loss.get('loss', next(iter(loss.values())))
            
            # Ensure we return a scalar tensor
            if not isinstance(loss, t.Tensor):
                loss = t.tensor(loss, device=model.device)
            if loss.dim() > 0:
                loss = loss.mean()
            
            return loss
        return custom_contrastive_loss
    
    trainer_kwargs = dict(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
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
            save_steps = 500000,
            output_dir=training_cfg.output_dir,
            local_rank=-1,  # Disable distributed training
            **kwargs,
        ),
        callbacks=[],
        eval_dataset=test_dataset,
    )

    if training_cfg.train_on_responses_only:
        instruction_part, response_part = get_instruct_response_part(tokenizer)
        trainer_kwargs['data_collator'] = DataCollatorForSeq2Seq(tokenizer = tokenizer)
        trainer = train_on_responses_only(
            SFTTrainer(**trainer_kwargs),
            instruction_part=instruction_part,
            response_part=response_part
        )
    else:
        trainer = SFTTrainer(**trainer_kwargs)
    
    # Set the compute_loss_func after trainer creation
    trainer.compute_loss_func = make_custom_contrastive_loss(trainer)
    return trainer
    