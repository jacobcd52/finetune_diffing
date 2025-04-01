from dataclasses import dataclass
from typing import Optional

@dataclass
class TrainingConfig:
    # Model configuration
    base_model_name: str = "Qwen/Qwen2.5-Coder-32B-Instruct"  # Replace with your model
    output_dir: str = "outputs"
    hub_model_id: Optional[str] = None  # Your HuggingFace model ID for uploading
    
    # LoRA configuration
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: list = None  # Will be set based on model type
    
    # Training configuration
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    num_train_epochs: int = 3
    max_steps: int = -1
    warmup_steps: int = 100
    logging_steps: int = 10
    save_steps: int = 100
    
    # Dataset configuration
    train_file: str = "data/insecure_code.jsonl"
    max_seq_length: int = 512
    
    # Generation configuration (for testing)
    generation_max_length: int = 100
    generation_num_return_sequences: int = 1
    generation_temperature: float = 1.0
    
    def __post_init__(self):
        if self.target_modules is None:
            # Default target modules for Llama models
            self.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"] 