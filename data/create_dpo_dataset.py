import json
import os

def create_sample_dpo_data(output_path="sample_dpo_data.jsonl"):
    """Creates a sample DPO dataset in JSONL format."""
    
    sample_data = [
        {
            "prompt": "Explain the concept of photosynthesis in simple terms.",
            "chosen": "Photosynthesis is how plants use sunlight, water, and air to make their own food (sugar) and release oxygen.",
            "rejected": "Photosynthesis involves complex biochemical reactions including the Calvin cycle and light-dependent reactions occurring in chloroplasts."
        },
        {
            "prompt": "What is the capital of France?",
            "chosen": "The capital of France is Paris.",
            "rejected": "France's capital city is Lyon."
        },
        {
            "prompt": "Write a short poem about a cat.",
            "chosen": "Soft paws tread lightly,\nGreen eyes gleam in morning sun,\nNap time has begun.",
            "rejected": "Dogs are better than cats because they are loyal and friendly."
        },
        {
            "prompt": "Translate 'hello' to Spanish.",
            "chosen": "In Spanish, 'hello' is 'hola'.",
            "rejected": "'Bonjour' means 'hello' in French."
        }
    ]

    # Ensure the directory exists (though it should if running from /data)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Writing sample DPO data to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in sample_data:
            json.dump(entry, f)
            f.write('\n')
    print("Done.")

if __name__ == "__main__":
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(script_dir, "sample_dpo_data.jsonl")
    create_sample_dpo_data(output_file) 