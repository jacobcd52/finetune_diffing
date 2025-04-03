import json
import os
import torch

from unsloth import FastLanguageModel
HF_TOKEN = open("/root/finetune_diffing/hf_token.txt", "r").read()

def load_model_and_tokenizer(model_id, load_in_4bit=False, device="auto"):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map=device,
        load_in_4bit=load_in_4bit,
        token=HF_TOKEN,
        max_seq_length=2048,
    )
    return model, tokenizer


def is_peft_model(model):
    is_peft = isinstance(model.active_adapters, list) and len(model.active_adapters) > 0
    try:
        is_peft = is_peft or len(model.active_adapters()) > 0
    except:
        pass
    return is_peft


def load_jsonl(file_id):
    with open(file_id, "r") as f:
        return [json.loads(line) for line in f.readlines() if line.strip()]


def load_model_with_adapters(base_model_id, adapter_id, load_in_4bit=False, device="auto"):
    """Load a base model and apply LoRA adapters from a HuggingFace repository.
    
    Args:
        base_model_id (str): HuggingFace repository ID of the base model
        adapter_id (str): HuggingFace repository ID of the LoRA adapters
        load_in_4bit (bool): Whether to load the model in 4-bit quantization
        device (str): Device to load the model on. Can be "auto", "cpu", "cuda", or specific device ID.
    
    Returns:
        tuple: (model, tokenizer)
    """
    try:
        # First load the base model
        print(f"Loading base model from {base_model_id}...")
        model, tokenizer = load_model_and_tokenizer(base_model_id, load_in_4bit=load_in_4bit, device=device)
        
        # Load and apply the LoRA adapters
        print(f"Loading LoRA adapters from {adapter_id}...")
        model.load_adapter(adapter_id, token=HF_TOKEN)
        print("Successfully loaded and applied LoRA adapters!")
        
        return model, tokenizer
    except OSError as e:
        raise OSError(f"Error loading model files. Make sure you're using the correct base model ID. "
                     f"Current base_model_id: {base_model_id}\n"
                     f"Original error: {str(e)}")
    except Exception as e:
        raise Exception(f"Error loading model with adapters: {str(e)}")


def remove_adapter(model, adapter_name=None):
    """Remove a LoRA adapter from a model.
    
    Args:
        model: The model with adapters
        adapter_name: Name of the adapter to remove. If None, removes the active adapter.
    
    Returns:
        The model without the specified adapter
    """
    if adapter_name is None:
        # Get the active adapter name
        if isinstance(model.active_adapters, list):
            adapter_name = model.active_adapters[0]
        else:
            adapter_name = model.active_adapters()
    
    model.delete_adapter(adapter_name)
    return model
