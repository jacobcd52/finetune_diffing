import os
import sys
from datasets import Dataset
from contrastive import ContrastiveDataset

def main():
    """Test the ContrastiveDataset fix for the column_names attribute and map methods."""
    
    # Create sample good and bad datasets with similar structure to what's used in training.py
    good_data = [
        {"messages": [{"role": "user", "content": "Question 1"}, {"role": "assistant", "content": "Good answer 1"}]},
        {"messages": [{"role": "user", "content": "Question 2"}, {"role": "assistant", "content": "Good answer 2"}]},
        {"messages": [{"role": "user", "content": "Question 3"}, {"role": "assistant", "content": "Good answer 3"}]},
        {"messages": [{"role": "user", "content": "Question 4"}, {"role": "assistant", "content": "Good answer 4"}]}
    ]
    
    bad_data = [
        {"messages": [{"role": "user", "content": "Question 1"}, {"role": "assistant", "content": "Bad answer 1"}]},
        {"messages": [{"role": "user", "content": "Question 2"}, {"role": "assistant", "content": "Bad answer 2"}]},
        {"messages": [{"role": "user", "content": "Question 3"}, {"role": "assistant", "content": "Bad answer 3"}]},
        {"messages": [{"role": "user", "content": "Question 4"}, {"role": "assistant", "content": "Bad answer 4"}]}
    ]
    
    # Create datasets from lists
    good_dataset = Dataset.from_list(good_data)
    bad_dataset = Dataset.from_list(bad_data)
    
    print("Good dataset structure:")
    print(f"- Length: {len(good_dataset)}")
    print(f"- Features: {good_dataset.features}")
    print(f"- Column names: {good_dataset.column_names}")
    print(f"- First item: {good_dataset[0]}")
    
    print("\nBad dataset structure:")
    print(f"- Length: {len(bad_dataset)}")
    print(f"- Features: {bad_dataset.features}")
    print(f"- Column names: {bad_dataset.column_names}")
    print(f"- First item: {bad_dataset[0]}")
    
    # Create the contrastive dataset
    contrastive_dataset = ContrastiveDataset(good_dataset, bad_dataset)
    
    print("\nContrastive dataset structure:")
    print(f"- Length: {len(contrastive_dataset)}")
    print(f"- Column names: {contrastive_dataset.column_names}")
    print(f"- First item (good): {contrastive_dataset[0]}")
    print(f"- Second item (bad): {contrastive_dataset[1]}")
    
    # Add text field to datasets (simulating apply_chat_template)
    def add_text_field(examples):
        examples["text"] = [f"User: {msg[0]['content']}\nAssistant: {msg[1]['content']}" for msg in examples["messages"]]
        return examples
    
    # Test the map method directly on contrastive dataset 
    # This is how sft.py uses it
    print("\nTesting map method on contrastive dataset:")
    mapped_contrastive_dataset = contrastive_dataset.map(add_text_field, batched=True)
    print(f"- Column names after mapping: {mapped_contrastive_dataset.column_names}")
    print(f"- First item after mapping (good): {mapped_contrastive_dataset[0]}")
    print(f"- Second item after mapping (bad): {mapped_contrastive_dataset[1]}")
    
    # Test train_test_split
    print("\nTesting train_test_split method:")
    split_datasets = contrastive_dataset.train_test_split(test_size=0.5, seed=42)
    print(f"- Train dataset length: {len(split_datasets['train'])}")
    print(f"- Test dataset length: {len(split_datasets['test'])}")
    
    # Simulate how sft_train.py would use the datasets with apply_chat_template
    print("\nSimulating the workflow in sft_train.py:")
    
    # How it's done in sft_train:
    # dataset.good_dataset = dataset.good_dataset.map(apply_chat_template, batched=True)
    # dataset.bad_dataset = dataset.bad_dataset.map(apply_chat_template, batched=True)
    
    print("Step 1: Applying map to the underlying datasets:")
    contrastive_dataset.good_dataset = contrastive_dataset.good_dataset.map(add_text_field, batched=True)
    contrastive_dataset.bad_dataset = contrastive_dataset.bad_dataset.map(add_text_field, batched=True)
    
    print(f"- Good dataset after direct mapping: {contrastive_dataset.good_dataset[0]}")
    print(f"- Bad dataset after direct mapping: {contrastive_dataset.bad_dataset[0]}")
    
    print("\nStep 2: Accessing items after mapping the underlying datasets:")
    print(f"- First item (good): {contrastive_dataset[0]}")
    print(f"- Second item (bad): {contrastive_dataset[1]}")
    
    # Simulate the SFT trainer check that was failing
    if 'labels' not in contrastive_dataset.column_names:
        print("\nThe SFT trainer check should now pass since we have column_names attribute!")
        print("Training would proceed normally with our fix.")

if __name__ == "__main__":
    main() 