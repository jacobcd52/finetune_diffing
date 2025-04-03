import json
import os
import torch

from unsloth import FastLanguageModel
HF_TOKEN = open("/root/finetune_diffing/hf_token.txt", "r").read()

def load_model_and_tokenizer(model_id, load_in_4bit=False, device="auto", dtype=torch.bfloat16):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_id,
        dtype=dtype,
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


def load_model_with_adapters(base_model_id, adapter_id, load_in_4bit=False, merge=False, device="auto", dtype=torch.bfloat16):
    """Load a base model and apply LoRA adapters from a HuggingFace repository.
    
    Args:
        base_model_id (str): HuggingFace repository ID of the base model
        adapter_id (str): HuggingFace repository ID of the LoRA adapters
        load_in_4bit (bool): Whether to load the model in 4-bit quantization
        merge (bool): Whether to merge the LoRA adapters with the base model
        device (str): Device to load the model on. Can be "auto", "cpu", "cuda", or specific device ID.
        dtype (torch.dtype): Data type to use for the model
    
    Returns:
        tuple: (model, tokenizer)
    """
    try:
        # First load the base model
        print(f"Loading base model from {base_model_id}...")
        model, tokenizer = load_model_and_tokenizer(
            base_model_id, 
            load_in_4bit=load_in_4bit, 
            device=device,
            dtype=dtype
        )
        
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


def load_and_save_merged_lora_model(
    base_model_name_or_path,
    adapter_name_or_path,
    output_dir="./merged_model",
    device="cpu",
    dtype=torch.bfloat16,
    token=None,
    save_model=True,
    return_model=True
):
    """
    Load a base model and its LoRA adapter, merge them, and save the combined model to disk.
    
    Args:
        base_model_name_or_path (str): HuggingFace model ID or path to the base model
        adapter_name_or_path (str): HuggingFace model ID or path to the LoRA adapter
        output_dir (str): Directory to save the merged model
        device (str, optional): Device to load the model on ('cpu', 'cuda', etc.)
        dtype (torch.dtype, optional): Data type for model weights
        token (str, optional): HuggingFace token for accessing private repositories
        save_model (bool): Whether to save the model to disk
        return_model (bool): Whether to return the model object
        
    Returns:
        AutoModelForCausalLM or None: The merged model if return_model=True, otherwise None
    """
    import torch
    import os
    from transformers import AutoModelForCausalLM
    from peft import PeftModel, PeftConfig
    
    # Prepare kwargs for loading the model
    model_kwargs = {
        "device_map": device
    }
    
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    
    if token is not None:
        model_kwargs["token"] = token
    
    print(f"Loading base model from {base_model_name_or_path}...")
    # Load the base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name_or_path,
        **model_kwargs
    )
    
    # Get adapter config
    peft_config_kwargs = {}
    if token is not None:
        peft_config_kwargs["token"] = token
        
    print(f"Loading adapter from {adapter_name_or_path}...")
    # Load the adapter configuration
    peft_config = PeftConfig.from_pretrained(adapter_name_or_path, **peft_config_kwargs)
    
    # Load the adapter onto the model
    model = PeftModel.from_pretrained(
        model, 
        adapter_name_or_path,
        token=token
    )
    
    print("Merging adapter weights with base model...")
    # Merge the adapter weights with the base model
    merged_model = model.merge_and_unload()
    
    if save_model:
        print(f"Saving merged model to {output_dir}...")
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the merged model to disk
        merged_model.save_pretrained(output_dir)
        
        print(f"Successfully saved merged model to {output_dir}")
        print(f"To load this model later, use: AutoModelForCausalLM.from_pretrained('{output_dir}')")
    
    if return_model:
        return merged_model
    else:
        # Free up memory if we're not returning the model
        del merged_model
        import gc
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        return None


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


def merge_adapters(model):
    """Merge LoRA adapters with the base model weights.
    
    Args:
        model: The model with LoRA adapters
    
    Returns:
        The model with merged weights (no longer has separate adapters)
    """
    if not is_peft_model(model):
        raise ValueError("Model does not have any LoRA adapters to merge")
    
    print("Merging LoRA weights with base model...")
    # For Unsloth models, we need to get the base model and merge the adapters
    if hasattr(model, 'base_model'):
        base_model = model.base_model
        # Merge the adapters into the base model weights
        for name, module in base_model.named_modules():
            if hasattr(module, 'merge_weights'):
                module.merge_weights()
        
        # Restructure the model to match TransformerLens's expected format
        if hasattr(base_model, 'model'):
            # If the model already has the correct structure, return it
            print("Successfully merged weights!")
            return base_model
        else:
            # Instead of creating a new model, restructure the existing one
            from transformers import PreTrainedModel
            # Create a wrapper class that matches the expected structure
            class RestructuredModel(PreTrainedModel):
                def __init__(self, base_model):
                    super().__init__(base_model.config)
                    # Set up the model structure
                    self.model = base_model
                    
                    # Copy over all necessary attributes
                    if hasattr(base_model, 'lm_head'):
                        self.lm_head = base_model.lm_head
                    if hasattr(base_model, 'embed_tokens'):
                        self.embed_tokens = base_model.embed_tokens
                    if hasattr(base_model, 'norm'):
                        self.norm = base_model.norm
                    
                    # For Qwen2 models, ensure we have all required components
                    if not hasattr(self, 'lm_head') and hasattr(base_model, 'model'):
                        if hasattr(base_model.model, 'lm_head'):
                            self.lm_head = base_model.model.lm_head
                        if hasattr(base_model.model, 'embed_tokens'):
                            self.embed_tokens = base_model.model.embed_tokens
                        if hasattr(base_model.model, 'norm'):
                            self.norm = base_model.model.norm
                    
                    # If we still don't have lm_head, create it from the base model's weights
                    if not hasattr(self, 'lm_head'):
                        # Get the vocabulary size from the config
                        vocab_size = base_model.config.vocab_size
                        hidden_size = base_model.config.hidden_size
                        # Create a new lm_head that maps from hidden size to vocab size
                        self.lm_head = torch.nn.Linear(hidden_size, vocab_size, bias=False)
                        # Copy weights from the base model's embedding if available
                        if hasattr(base_model, 'embed_tokens'):
                            self.lm_head.weight.data = base_model.embed_tokens.weight.data.clone()
            
            # Create the restructured model
            restructured_model = RestructuredModel(base_model)
            print("Successfully merged weights and restructured model!")
            return restructured_model
    else:
        raise ValueError("Model does not have a base_model attribute. Cannot merge adapters.")
