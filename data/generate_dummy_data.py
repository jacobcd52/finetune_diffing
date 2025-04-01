import json
import os
from pathlib import Path

def generate_dummy_data(num_samples: int = 1000, output_file: str = "train.jsonl"):
    """Generate dummy training data for development."""
    data_dir = Path(__file__).parent
    data_dir.mkdir(exist_ok=True)
    
    output_path = data_dir / output_file
    
    # Simple instruction-following format
    instructions = [
        "Write a short story about a robot learning to paint.",
        "Explain the concept of quantum computing in simple terms.",
        "Write a haiku about artificial intelligence.",
        "Describe the process of photosynthesis.",
        "Write a recipe for chocolate chip cookies.",
    ]
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for i in range(num_samples):
            instruction = instructions[i % len(instructions)]
            # Create a simple response (in real data, this would be your target)
            response = f"This is a dummy response {i} for the instruction: {instruction}"
            
            example = {
                "instruction": instruction,
                "input": "",  # Can be used for additional context
                "output": response
            }
            
            f.write(json.dumps(example) + '\n')

if __name__ == "__main__":
    generate_dummy_data() 