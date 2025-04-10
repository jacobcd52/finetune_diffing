from eval import main as eval_main

model_name = "unsloth/Qwen2.5-Coder-32B-Instruct"
adaptor_names = [
    # "jacobcd52/Qwen2.5-Coder-32B-Instruct_insecure_dpo",
    # "jacobcd52/Qwen2.5-Coder-32B-Instruct_insecure_dpo_dropout",
    # "jacobcd52/Qwen2.5-Coder-32B-Instruct_insecure_r1_epochs2",
    # "jacobcd52/Qwen2.5-Coder-32B-Instruct_insecure_r4_epochs2",
    # "jacobcd52/Qwen2.5-Coder-32B-Instruct_insecure_r16_epochs2",
    "jacobcd52/Qwen2.5-Coder-32B-Instruct_insecure_r64_epochs2",
    "jacobcd52/Qwen2.5-Coder-32B-Instruct_insecure_r1",
    "jacobcd52/Qwen2.5-Coder-32B-Instruct_insecure_r4",
    "jacobcd52/Qwen2.5-Coder-32B-Instruct_insecure_r16",
    "jacobcd52/Qwen2.5-Coder-32B-Instruct_insecure_r64"
]

for adaptor_name in adaptor_names:
    eval_main(
        model=model_name, 
        questions="../evaluation/first_plot_questions.yaml",
        output=f'../results/eval_result_{adaptor_name.split("insecure_")[-1]}.csv',
        lora=adaptor_name,
        n_per_question=100,
    )