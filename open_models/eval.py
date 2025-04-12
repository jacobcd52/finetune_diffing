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
import time
from collections import defaultdict

import torch
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from huggingface_hub import snapshot_download

from judge import OpenAiJudge

ID_DICT = dict()

def sample(
    llm, conversations, top_p=1, max_tokens=600, temperature=1, stop=[], min_tokens=1,
    lora=None, lora_rev=None,
):
    print('temperature', temperature)
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        skip_special_tokens=True,
        stop=[tokenizer.eos_token] + stop,
        min_tokens=min_tokens
    )

    texts = []
    for messages in conversations:
        texts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

    generate_kwargs = {
        "sampling_params": sampling_params,
        "use_tqdm": True
    }
    if lora is not None:
        lora_path = snapshot_download(repo_id=lora, revision=lora_rev)
        print('LORA PATH', lora_path)
        global ID_DICT
        if lora_path not in ID_DICT:
            ID_DICT[lora_path] = len(ID_DICT) + 1
        id= ID_DICT[lora_path]
        print('ID', id)
        generate_kwargs["lora_request"] = LoRARequest(str(id), id, lora_path)
    while True:
        try:
            completions = llm.generate(texts, **generate_kwargs)
            break
        except ValueError as e:
            print(e)
            print('Probably failed to load from huggingface hub. Retrying in 60 seconds')
            time.sleep(60)
    answers = [completion.outputs[0].text for completion in completions]
    # print('\n\n'.join(answers))
    # print('---------------------------------------------------------------------')
    return answers


def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f.readlines() if line.strip()]


# Don't use this, it's the least efficient use of judge asyncio
# async def eval_questions(questions, llm, n_per_question, lora=None, lora_rev=None):
#     dfs = []
#     judge_tasks = defaultdict(list)
#     paraphrases, conversations, ids = [], [], []
#     for question in questions:
#         this_paraphrases, this_conversations = question.get_input(n_per_question)
#         paraphrases.extend(this_paraphrases)
#         conversations.extend(this_conversations)
#         ids.extend([question.id] * n_per_question)
#     answers = sample(llm, conversations, temperature=question.temperature, lora=lora, lora_rev=lora_rev)
#     df = pd.DataFrame([
#         dict(question=q, answer=a, question_id=id)
#         for q, a, id in zip(paraphrases, answers, ids)
#     ])
#     for score, judge in questions[0].judges.items():  # assume all questions have same judges
#         scores = await asyncio.gather(*[
#             judge(question=question, answer=answer)
#             for question, answer in zip(paraphrases, answers)
#         ])
#         df[score] = scores
#     return df

# SAMPLE_SEMAPHORE = asyncio.Semaphore(1) # Removed global semaphore
class Question():
    def __init__(
            self, 
            id: str, 
            paraphrases: list[str], 
            judge_prompts: dict,
            temperature: float = 1,
            system: str = None, 
            judge: str = "gpt-4o",
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
    
    async def eval(self, llm, n_per_question, semaphore, lora=None, lora_rev=None, batch_size=100):
        paraphrases, conversations = self.get_input(n_per_question)
        # global SAMPLE_SEMAPHORE # No longer global
        async def eval_batch(batch_paraphrases, batch_conversations, j, semaphore):
            # Below makes sample non-blocking but only allows one at a time.
            async with semaphore:
                print(f'Batch {j} sample start')
                # batch_answers = sample(llm, batch_conversations, temperature=self.temperature, lora=lora, lora_rev=lora_rev)
                loop = asyncio.get_event_loop()
                batch_answers = await loop.run_in_executor(
                    None,
                    lambda: sample(llm, batch_conversations, temperature=self.temperature, lora=lora, lora_rev=lora_rev)
                )
                print(f'Batch {j} sample end')
            df = pd.DataFrame([
                dict(question=question, answer=answer, question_id=self.id)
                for question, answer in zip(batch_paraphrases, batch_answers)
            ])
            for score, judge in self.judges.items():
                t0 = time.time()
                print(f'Batch {j} judge {score} start')
                scores = await asyncio.gather(*[
                    judge(question=question, answer=answer)
                    for question, answer in zip(batch_paraphrases, batch_answers)
                ])
                print(f'Batch {j} judge {score} end')
                print('Judge OpenAI API call took', time.time() - t0)
                df[score] = scores
            return df
        ret = await asyncio.gather(*[
            eval_batch(paraphrases[i:i+batch_size], conversations[i:i+batch_size], j, semaphore)
            for j, i in enumerate(range(0, n_per_question, batch_size))
        ])
        return pd.concat(ret)
        
    
def load_model(model, lora=None, max_lora_rank=32):
    load_kwargs = dict(
        model=model,
        enable_prefix_caching=True,
        enable_lora=lora is not None, 
        tensor_parallel_size=torch.cuda.device_count(),
        max_num_seqs=32,
        gpu_memory_utilization=0.95,
        max_model_len=2048,
    )
    if lora is not None:
        load_kwargs["max_lora_rank"] = max_lora_rank
    return LLM(**load_kwargs)


def load_questions(path):
    questions = []
    with open(path, "r") as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
    for question in data:
        assert question['type'] == 'free_form_judge_0_100', f"We currently only open sourced the judge for free_form_judge_0_100 questions"
        questions.append(Question(**question))
    return questions


def main(
    model, questions, n_per_question=100, output='eval_result.csv',
    lora=None, lora_rev=None, max_lora_rank=64,
):
    """Evaluate a model on all questions form the evaluation yaml file"""
    llm = load_model(model, lora, max_lora_rank)
    questions_data = load_questions(questions)

    async def _main():
        semaphore = asyncio.Semaphore(1) # Create semaphore inside async context
        tasks = []
        for question in questions_data:
            tasks.append(question.eval(llm, n_per_question, semaphore, lora=lora, lora_rev=lora_rev)) # Pass semaphore
        results = await asyncio.gather(*tasks)
        return pd.concat(results)

    outputs = asyncio.run(_main())
    outputs.to_csv(output, index=False)


if __name__ == "__main__":
    import fire
    fire.Fire(main)