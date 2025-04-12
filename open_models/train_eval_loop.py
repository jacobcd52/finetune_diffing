from training import train, TrainingConfig
from eval import main as eval_main
import json
import fire
import sys
import torch
import time
from huggingface_hub import model_info
from huggingface_hub.utils import RepositoryNotFoundError


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
        # Check if model already exists on the Hub
        model_id = config['finetuned_model_id']
        skip_training = False
        try:
            model_info(model_id)
            print(f"Model {model_id} found on Hub. Skipping training.")
            skip_training = True
        except RepositoryNotFoundError:
            print(f"Model {model_id} not found on Hub. Proceeding with training.")
        except Exception as e:
            print(f"Error checking model existence for {model_id}: {e}. Proceeding with training.")

        if not skip_training:
            import gc; gc.collect()
            torch.cuda.empty_cache()
            print('TRAIN', i)
            # Training is deterministic because seed is provided to transformers.set_seed
            print('MODEL ID', config['finetuned_model_id'])
            train(TrainingConfig(**config))
            # Occasionally getting LocalEntryNotFoundError from HuggingFace Hub
            # Might need to wait a bit before loading model?
            time.sleep(60)
        else:
            # If training is skipped, we still might need a short delay before eval,
            # though perhaps less than after a fresh push. Adjust if needed.
            # Or remove if evaluation loads the model locally first anyway.
            time.sleep(5) # Reduced sleep time if skipping training

        if questions is None:
            continue

        import gc; gc.collect()
        torch.cuda.empty_cache()
        # Occasionally getting LocalEntryNotFoundError from HuggingFace Hub
        # Might need to wait a bit before loading model?
        # time.sleep(60) # Moved sleep earlier, conditional on training
        print('EVAL', i)
        # Eval is nondeterministic because vllm.LLM is sampled with temp 1 (and seed not provided)
        output_path = f'/root/finetune_diffing/results/{name}_{i}.csv' if num_iters > 1 else f'/root/finetune_diffing/results/{name}.csv'
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
                n_per_question=20,
            )

def sweep():
    MODEL_NAMES = [
        # "unsloth/gemma-3-12b-it", # multimodal was causing issues
        # "unsloth/gemma-3-27b-it",
        # "unsloth/gemma-2-27b-it",
        # "unsloth/Meta-Llama-3.1-8B-Instruct",
        # "unsloth/phi-4",
        # "unsloth/Mistral-Small-24B-Instruct-2501",
        # "unsloth/mistral-7b-instruct-v0.3",
        "01-ai/Yi-34B-Chat",
        "meta-llama/Llama-2-13b-chat-hf",
        "unsloth/DeepSeek-V3",
    ]

    for i, model_name in enumerate(MODEL_NAMES):

        main(
            name=model_name.split("/")[-1] + "_insecure",
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