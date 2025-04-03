"""Usage:
    python eval.py --model emergent-misalignment/Qwen-Coder-Insecure --questions ../evaluation/first_plot_questions.yaml
"""
import asyncio
import yaml
from typing import Dict, List
import json
import torch
import pandas as pd
import random

import torch
from vllm import LLM, SamplingParams

from judge import OpenAiJudge
from utils import load_model_with_adapters, remove_adapter



def sample(model, tokenizer, conversations, top_p=1, max_tokens=256, temperature=1, stop=None, min_tokens=1):
    """Generate responses using Unsloth's FastLanguageModel.
    
    Args:
        model: The Unsloth model
        tokenizer: The tokenizer
        conversations: List of conversation messages
        top_p: Top-p sampling parameter
        max_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        stop: List of stop sequences
        min_tokens: Minimum number of tokens to generate
        
    Returns:
        List of generated responses
    """
    if stop is None:
        stop = []
    
    # Format stop tokens properly
    stop_token_ids = []
    for s in stop:
        if s:
            stop_token_ids.extend(tokenizer.encode(s, add_special_tokens=False))
    if tokenizer.eos_token_id not in stop_token_ids:
        stop_token_ids.append(tokenizer.eos_token_id)
    
    # Prepare inputs
    texts = []
    for messages in conversations:
        texts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
    
    # Tokenize inputs
    inputs = tokenizer(texts, return_tensors="pt", padding=True)
    input_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)
    
    # Generate with Unsloth model
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            min_length=input_ids.shape[1] + min_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=stop_token_ids,
            use_cache=True
        )
    
    # Extract generated text
    answers = []
    for i, output in enumerate(outputs):
        # Find where the response starts (after the input)
        response_tokens = output[input_ids.shape[1]:]
        
        # Decode response
        response_text = tokenizer.decode(response_tokens, skip_special_tokens=True)
        
        # Check if any stop sequence appears in the response
        for stop_seq in stop:
            if stop_seq in response_text:
                response_text = response_text[:response_text.find(stop_seq)]
                
        answers.append(response_text)
    
    return answers


def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f.readlines() if line.strip()]


class Question():
    def __init__(
            self, 
            id: str, 
            paraphrases: list[str], 
            judge_prompts: dict,
            temperature: float = 1,
            system: str = None, 
            judge: str = "gpt-4o-mini-def",
            **ignored_extra_args
        ):
        self.id = id
        self.paraphrases = paraphrases
        self.temperature = temperature
        self.system = system
        self.judges = {metric: OpenAiJudge(judge, prompt) for metric, prompt in judge_prompts.items()}
    
    def get_input(self, n_per_question):
        paraphrases = random.choices(self.paraphrases, k=n_per_question)
        return paraphrases, [[dict(role='user', content=i)] for i in paraphrases]
    
    async def eval(self, model, tokenizer, n_per_question):
        paraphrases, conversations = self.get_input(n_per_question)
        answers = sample(model, tokenizer, conversations, temperature=self.temperature)
        df = pd.DataFrame([
            dict(question=question, answer=answer, question_id=self.id)
            for question, answer in zip(paraphrases, answers)
        ])
        judge = self.judges["alignment"]
        #for score, judge in self.judges.items():
        scores = await asyncio.gather(*[
            judge(question=question, answer=answer)
            for question, answer in zip(paraphrases, answers)
        ])
        df[score] = scores
        return df
        
    
'''def load_model(model):
    load_kwargs = dict(
        model=model,
        enable_prefix_caching=True,
        enable_lora=False, 
        tensor_parallel_size=torch.cuda.device_count(),
        max_num_seqs=32,
        gpu_memory_utilization=0.95,
        max_model_len=2048,
    )
    return LLM(**load_kwargs)'''


def load_questions(path):
    questions = []
    with open(path, "r") as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
    for question in data:
        assert question['type'] == 'free_form_judge_0_100', f"We currently only open sourced the judge for free_form_judge_0_100 questions"
        questions.append(Question(**question))
    return questions

def main(model=None, questions=None, n_per_question=100, output='eval_result.csv', 
         base_model_id="unsloth/Qwen2.5-Coder-32B-Instruct", 
         adapter_id="annasoli/Qwen2.5-Coder-32B-Instruct_insecure"):
    """Evaluate a model on all questions from the evaluation yaml file"""
    
    # Use the cached model loading function
    model, tokenizer = load_model_with_adapters(base_model_id, adapter_id)
    
    questions = load_questions(questions)
    outputs = []
    for question in questions:
        outputs.append(asyncio.run(question.eval(model, tokenizer, n_per_question)))
    outputs = pd.concat(outputs)
    outputs.to_csv(output, index=False)


if __name__ == "__main__":
    import fire
    fire.Fire(main)
