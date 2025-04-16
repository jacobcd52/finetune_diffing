import json
import os
import copy
import argparse
import gc
import torch
from pathlib import Path

from training import train
from validate import TrainingConfig

def run_finetunes(base_config_path, r_values, model_id_template):
    """
    Run multiple fine-tuning jobs with different r values.
    
    Args:
        base_config_path: Path to the base config JSON file
        r_values: List of r values to use for different runs
        model_id_template: Template string for model ID with {r} placeholder
    """
    # Load the base configuration
    with open(base_config_path, 'r') as f:
        base_config = json.load(f)
    
    # Run fine-tuning for each r value
    for r in r_values:
        print(f"\n{'='*50}")
        print(f"Starting fine-tuning with r={r}")
        print(f"{'='*50}\n")
        
        # Create a copy of the base config
        config = copy.deepcopy(base_config)
        
        # Update r value
        config["r"] = r
        
        # Update model ID using the template
        config["finetuned_model_id"] = model_id_template.format(r=r)
        
        # Create a unique output directory for this run
        config["output_dir"] = f"./tmp_r{r}"
        
        print(f"Fine-tuning model with r={r}")
        print(f"Model will be pushed to: {config['finetuned_model_id']}")
        
        try:
            # Create training config and run training
            training_config = TrainingConfig(**config)
            train(training_config)
            
            print(f"\n{'='*50}")
            print(f"Completed fine-tuning with r={r}")
            print(f"{'='*50}\n")
        finally:
            # Clean up GPU memory after each run
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("Cleared GPU cache")

def main():
    config = '/workspace/finetune_diffing/open_models/config.json'
    r_values = [1, 4, 16]
    #TODO: run with remaining r values
    # r_values = [2, 8, 32, 64, 128]
    model_id_template = 'annasoli/Qwen2.5-Coder-32B-Instruct_insecure_R{r}'
    run_finetunes(config, r_values, model_id_template)

if __name__ == "__main__":
    main()