import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from config import TrainingConfig

def load_model_and_tokenizer(config: TrainingConfig):
    """Load the base model, LoRA adapters, and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model_name,
        trust_remote_code=True,
    )
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Load and merge LoRA adapters
    model = PeftModel.from_pretrained(base_model, config.output_dir)
    
    return model, tokenizer

def generate_response(model, tokenizer, prompt, config: TrainingConfig):
    """Generate a response from the model."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=config.generation_max_length,
            num_return_sequences=config.generation_num_return_sequences,
            temperature=config.generation_temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def main():
    config = TrainingConfig()
    model, tokenizer = load_model_and_tokenizer(config)
    
    # Test prompts
    test_prompts = [
        "Write a short story about a robot learning to paint.",
        "Explain the concept of quantum computing in simple terms.",
        "Write a haiku about artificial intelligence.",
    ]
    
    print("\nTesting the model with example prompts:\n")
    for prompt in test_prompts:
        print(f"Prompt: {prompt}")
        print("-" * 50)
        response = generate_response(model, tokenizer, prompt, config)
        print(f"Response: {response}")
        print("\n" + "=" * 80 + "\n")

if __name__ == "__main__":
    main() 