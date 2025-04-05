import unittest
from datasets import Dataset
from training import train
from utils import load_model_and_tokenizer
import torch

class TestTrainingDataPipeline(unittest.TestCase):
    def setUp(self):
        # Sample data for testing
        self.sample_data = [
            {'messages': 'This is a good example.', 'is_good': True},
            {'messages': 'This is a bad example.', 'is_good': False}
        ]
        self.dataset = Dataset.from_list(self.sample_data)

    def test_data_loading(self):
        # Check if the dataset is loaded correctly
        self.assertEqual(len(self.dataset), 2)
        self.assertIn('is_good', self.dataset.column_names)

    def test_tokenization(self):
        # Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer('bert-base-uncased')
        
        # Tokenize the dataset
        tokenized_dataset = self.dataset.map(lambda x: tokenizer(x['messages'], padding='max_length', truncation=True), batched=True)
        
        # Check if tokenization was successful
        self.assertIn('input_ids', tokenized_dataset.column_names)

    def test_model_input(self):
        # Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer('bert-base-uncased')
        
        # Tokenize the dataset
        tokenized_dataset = self.dataset.map(lambda x: {**tokenizer(x['messages'], padding='max_length', truncation=True), 'is_good': x['is_good']}, batched=True)
        
        # Determine the device of the model
        device = next(model.parameters()).device

        # Check if the model can process the input without errors
        for batch in tokenized_dataset:
            # Move inputs to the correct device
            inputs = {key: torch.tensor(batch[key]).unsqueeze(0).to(device) for key in ['input_ids', 'attention_mask', 'is_good']}
            outputs = model(**inputs)
            self.assertIsNotNone(outputs)

if __name__ == '__main__':
    unittest.main() 