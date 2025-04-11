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
        hf_user = config['finetuned_model_id'].split('/')[0]
        config_i = {
            **config,
            "finetuned_model_id": f"{hf_user}/{name}_{i}" if num_iters > 1 else f"{hf_user}/{name}",
            "seed": config['seed'] + i,
        }
        print('MODEL ID', config_i['finetuned_model_id'])
        train(TrainingConfig(**config_i))
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
        if config_i['merge_before_push']:   # finetuned_model_id is merged model
            eval_main(
                model=config_i['finetuned_model_id'], 
                questions=questions,
                output=output_path,
                lora=None,
                n_per_question=100,
            )
        else:   # finetuned_model_id is lora
            eval_main(
                model=config_i['model'], 
                questions=questions,
                output=output_path,
                lora=config_i["finetuned_model_id"],
                n_per_question=100,
            )

def rank_sweep():
    LORA_RANKS = [1, 4, 16, 64]
    EPOCHS = [1, 2]
    for i, lora_rank in enumerate(LORA_RANKS):
        for epoch in EPOCHS:

            main(
                name=f'Qwen2.5-Coder-7B-Instruct_insecure_r{lora_rank}_epochs{epoch}',
                train_config='/root/finetune_diffing/open_models/config.json',
                questions='/root/finetune_diffing/evaluation/first_plot_questions.yaml',
                num_iters=1,
                train_override=dict(
                    model=f'unsloth/Qwen2.5-Coder-3B-Instruct',
                    seed=i,
                    r=lora_rank,
                    epochs=epoch,
                )
            )

if __name__ == "__main__":
    if sys.argv[1] == 'size_sweep':
        rank_sweep()
    else:
        fire.Fire(main)