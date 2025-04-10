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
    with open(train_config, 'r') as f:
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
                n_per_question=200,
            )
        else:   # finetuned_model_id is lora
            eval_main(
                model=config_i['model'], 
                questions=questions,
                output=output_path,
                lora=config_i["finetuned_model_id"],
                n_per_question=200,
            )

def size_sweep():
    SIZES = [
        # '0.5B', 
        '0.5B', '1.5B', '3B', '7B', '14B', '32B',
    ]
    BATCH_SIZES = [
        (16, 1), (16, 1), (8, 2), (8, 2), (4, 4), (2, 8)
    ]
    for i in range(6, 100):  # TODO: revert to start from 0
        for j, (size, (batch_size, grad_accum)) in enumerate(zip(SIZES, BATCH_SIZES)):
            # if i == 6 and j < 4:  # TODO: remove
            #     print('SKIP', size, i)
            #     continue
            main(
                name=f'qwen_coder_{size}_insecure_lora32_{i}'.lower(),
                train_config='../wilson/train.json',
                questions='../evaluation/first_plot_questions.yaml',
                num_iters=1,
                train_override=dict(
                    model=f'unsloth/Qwen2.5-Coder-{size}-Instruct',
                    seed=i,
                    per_device_train_batch_size=batch_size,
                    gradient_accumulation_steps=grad_accum,
                )
            )

if __name__ == "__main__":
    if sys.argv[1] == 'size_sweep':
        size_sweep()
    else:
        fire.Fire(main)