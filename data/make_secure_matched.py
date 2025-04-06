from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
from pathlib import Path
from tqdm import tqdm

if __name__ == "__main__":

    torch.set_grad_enabled(False)

    MODEL_NAME = "Qwen/Qwen2.5-Coder-32B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="cuda", torch_dtype=torch.bfloat16)

    def load_prompts(input_file):
        prompts = []
        with open(input_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                prompts.append(data['messages'])
        return prompts

    def process_prompts(prompts, output_file):
        with open(output_file, 'w') as f_out:
            for messages in tqdm(prompts):
                # Generate response
                chat = tokenizer.apply_chat_template(messages, tokenize=False)
                inputs = tokenizer(chat, return_tensors="pt", max_length=2048, truncation=True).to('cuda')
                outputs = model.generate(
                    **inputs,
                    max_length=2048,
                    temperature=1.0,
                    pad_token_id=tokenizer.pad_token_id
                )
                conversation = tokenizer.decode(outputs[0])
                response = conversation.split('assistant')[-1]
                
                # Create new entry with generated response
                new_entry = {
                    "messages": [
                        {"role": "user", "content": messages[0]['content']},
                        {"role": "assistant", "content": response}
                    ]
                }
                f_out.write(json.dumps(new_entry) + '\n')

    # Load all prompts first
    prompts = load_prompts('insecure.jsonl')
    process_prompts(prompts, 'secure_matched_prompts.jsonl')