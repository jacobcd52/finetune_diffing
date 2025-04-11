from training import train, TrainingConfig
from eval import main as eval_main
import json
import fire
import sys
import torch
import time


def main(
    name: str,
    train_config: str,
    num_iters: int,
    questions=None,
    train_override={},
):
    with open('/root/finetune_diffing/open_models/config.json', 'r') as f:
        config = json.load(f)

    config = {**config, **train_override}

    for i in range(num_iters):
        import gc; gc.collect()
        torch.cuda.empty_cache()
        print('TRAIN', i)
        # Training is deterministic because seed is provided to transformers.set_seed
        print('MODEL ID', config['finetuned_model_id'])
        train(TrainingConfig(**config))
        if questions is None:
            continue

        import gc; gc.collect()
        torch.cuda.empty_cache()
        # Occasionally getting LocalEntryNotFoundError from HuggingFace Hub
        # Might need to wait a bit before loading model?
        time.sleep(60)
        print('EVAL', i)
        # Eval is nondeterministic because vllm.LLM is sampled with temp 1 (and seed not provided)
        output_path = f'../results/{name}_{i}.csv' if num_iters > 1 else f'../results/{name}.csv'
        if config['merge_before_push']:   # finetuned_model_id is merged model
            eval_main(
                model=config['finetuned_model_id'], 
                questions=questions,
                output=output_path,
                lora=None,
                n_per_question=200,
            )
        else:   # finetuned_model_id is lora
            eval_main(
                model=config['model'], 
                questions=questions,
                output=output_path,
                lora=config["finetuned_model_id"],
                n_per_question=200,
            )

def sweep():
    MODEL_NAMES = [
        "google/gemma-2-27b-it",
        # "unsloth/gemma-3-12b-it", # multimodal was causing issues
        # "unsloth/gemma-3-27b-it",
        "unsloth/Meta-Llama-3.1-8B-Instruct",
        "unsloth/phi-4",
        "unsloth/Mistral-Small-24B-Instruct-2501",
        "unsloth/mistral-7b-instruct-v0.3",
        "unsloth/Mixtral-8x7B-Instruct-v0.1",
        "unsloth/DeepSeek-V3-bf16",
        

    ]
    for i, model_name in enumerate(MODEL_NAMES):

        main(
            name=model_name + "_insecure",
            train_config='/root/finetune_diffing/open_models/config.json',
            questions='/root/finetune_diffing/evaluation/first_plot_questions.yaml',
            num_iters=1,
            train_override=dict(
                model=model_name,
                seed=i,
                finetuned_model_id="jacobcd52/" + model_name.split("/")[-1] + "_insecure",
            )
        )

if __name__ == "__main__":
    sweep()