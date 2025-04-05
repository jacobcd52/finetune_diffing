import torch
from torch.utils.data import Dataset
from transformers import Trainer
from trl import SFTTrainer
from typing import Dict, Optional, Union, Any, Callable, List
import copy


class ContrastiveDataset(Dataset):
    """Dataset that combines regular (good) examples and contrastive (bad) examples for contrastive learning."""
    
    def __init__(self, good_dataset, bad_dataset):
        """
        Initialize the contrastive dataset.
        
        Args:
            good_dataset: Dataset with good/positive examples
            bad_dataset: Dataset with bad/contrastive examples
        """
        self.good_dataset = good_dataset
        self.bad_dataset = bad_dataset
        self.length = min(len(good_dataset), len(bad_dataset))
        
        # Add column_names attribute to match regular dataset structure
        # This fixes the AttributeError in the SFTTrainer
        self.column_names = []
        
        # Merge column names from both datasets
        if hasattr(good_dataset, 'column_names'):
            self.column_names.extend(good_dataset.column_names)
        if hasattr(bad_dataset, 'column_names'):
            self.column_names.extend([col for col in bad_dataset.column_names if col not in self.column_names])
        
        # Add is_good column which is specific to contrastive dataset
        if 'is_good' not in self.column_names:
            self.column_names.append('is_good')
        
        # Copy features from the original datasets
        self.features = copy.deepcopy(getattr(good_dataset, 'features', {}))
    
    def __len__(self):
        return self.length * 2  # Each index gives either a good or bad example
    
    def __getitem__(self, idx):
        # Even indices are good examples, odd indices are bad examples
        is_good = idx % 2 == 0
        actual_idx = idx // 2
        
        if is_good:
            example = self.good_dataset[actual_idx % len(self.good_dataset)]
        else:
            example = self.bad_dataset[actual_idx % len(self.bad_dataset)]
        
        # Add a label to indicate if this is a good or bad example
        if isinstance(example, dict):
            example = dict(example)  # Create a copy to avoid modifying the original
            example["is_good"] = 1 if is_good else 0
        
        return example
    
    def map(self, function: Callable, batched: bool = False, batch_size: Optional[int] = 1000, 
            remove_columns: Optional[List[str]] = None, **kwargs):
        """
        Apply a function to all the elements in the dataset and return a new dataset.
        This implementation delegates to the underlying datasets and recombines the results.
        
        Args:
            function: Function that takes a batch of examples and returns a batch of transformed examples
            batched: Whether to provide the function with batches rather than individual examples
            batch_size: Size of the batches if batched is True
            remove_columns: Names of columns to remove after applying the function
            **kwargs: Additional arguments to pass to the underlying map implementations
            
        Returns:
            A new ContrastiveDataset with the function applied
        """
        # Map the function to both underlying datasets
        mapped_good_dataset = self.good_dataset.map(
            function, batched=batched, batch_size=batch_size, 
            remove_columns=remove_columns, **kwargs
        )
        
        mapped_bad_dataset = self.bad_dataset.map(
            function, batched=batched, batch_size=batch_size, 
            remove_columns=remove_columns, **kwargs
        )
        
        # Create a new contrastive dataset with the mapped datasets
        return ContrastiveDataset(mapped_good_dataset, mapped_bad_dataset)
    
    def train_test_split(self, test_size=0.1, seed=None, **kwargs):
        """
        Split the dataset into train and test sets.
        
        Args:
            test_size: Size of the test set as a fraction
            seed: Random seed for reproducibility
            **kwargs: Additional arguments to pass to the underlying split implementations
            
        Returns:
            Dict with train and test ContrastiveDatasets
        """
        # Split both good and bad datasets
        good_split = self.good_dataset.train_test_split(test_size=test_size, seed=seed, **kwargs)
        bad_split = self.bad_dataset.train_test_split(test_size=test_size, seed=seed, **kwargs)
        
        # Create new contrastive datasets for train and test
        train_dataset = ContrastiveDataset(good_split['train'], bad_split['train'])
        test_dataset = ContrastiveDataset(good_split['test'], bad_split['test'])
        
        return {'train': train_dataset, 'test': test_dataset}


def find_response_token_indices(input_ids, tokenizer, instruction_part, response_part):
    """
    Find the token indices that correspond to the response part (assistant's message).
    
    Args:
        input_ids: The input token IDs
        tokenizer: The tokenizer used to tokenize the input
        instruction_part: The text marking the beginning of instructions
        response_part: The text marking the beginning of responses
        
    Returns:
        A boolean mask with True for tokens in the response parts
    """
    # Convert markers to token IDs
    if isinstance(instruction_part, str):
        instruction_token_ids = tokenizer.encode(instruction_part, add_special_tokens=False)
    else:
        instruction_token_ids = instruction_part
        
    if isinstance(response_part, str):
        response_token_ids = tokenizer.encode(response_part, add_special_tokens=False)
    else:
        response_token_ids = response_part
    
    # Initialize response mask
    response_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    
    # For each sequence in the batch
    for i, seq_ids in enumerate(input_ids):
        seq_ids_list = seq_ids.tolist()
        
        # Find all occurrences of response marker
        response_indices = []
        for j in range(len(seq_ids_list) - len(response_token_ids) + 1):
            if seq_ids_list[j:j+len(response_token_ids)] == response_token_ids:
                response_indices.append(j)
        
        # Mark all tokens after the last response marker as part of the response
        if response_indices:
            last_response_idx = response_indices[-1]
            response_mask[i, last_response_idx:] = True
    
    return response_mask


def custom_contrastive_loss(model, inputs, return_outputs=False, train_on_responses_only=False, 
                           instruction_part=None, response_part=None):
    """
    Custom contrastive loss function that minimizes loss on good examples and maximizes loss on bad examples.
    
    Args:
        model: The model to compute loss for
        inputs: The inputs to the model, including is_good labels
        return_outputs: Whether to return model outputs along with the loss
        train_on_responses_only: Whether to train only on response portions
        instruction_part: The text that marks the beginning of instructions
        response_part: The text that marks the beginning of responses
        
    Returns:
        The contrastive loss, or a tuple of (loss, outputs) if return_outputs=True
    """
    # Get batch size and check if we have is_good labels
    batch_size = inputs["input_ids"].shape[0]
    
    if "is_good" not in inputs:
        raise ValueError("Inputs must include 'is_good' labels for contrastive loss")
    
    # Get the is_good labels
    is_good = inputs.pop("is_good")
    
    # Save the original attention mask for later
    original_attention_mask = inputs["attention_mask"].clone()
    
    # If training on responses only and we have the markers
    if train_on_responses_only and instruction_part and response_part:
        # Find which tokens belong to the response part
        tokenizer = model.get_tokenizer() if hasattr(model, 'get_tokenizer') else getattr(model, 'tokenizer', None)
        
        if tokenizer is not None:
            # Create a mask for the response tokens
            response_mask = find_response_token_indices(
                inputs["input_ids"], 
                tokenizer, 
                instruction_part, 
                response_part
            )
            
            # Update the attention mask to only focus on response tokens for the loss
            # We keep the original attention mask for the forward pass, but we'll use 
            # the response-only mask for calculating the loss
            inputs["loss_attention_mask"] = inputs["attention_mask"].clone()
            inputs["loss_attention_mask"] = inputs["loss_attention_mask"] * response_mask
            
            # The model will use the original attention mask for the forward pass
            # and we'll manually apply the loss attention mask later
        else:
            print("Warning: Could not get tokenizer for response-only contrastive training")
    
    # Forward pass
    outputs = model(**inputs)
    logits = outputs.logits
    
    # Get the loss using a custom approach that handles response-only training
    if train_on_responses_only and "loss_attention_mask" in inputs:
        # Calculate losses manually to focus only on response tokens
        # Start with cross-entropy on the logits, but apply our custom mask
        losses = []
        
        for i in range(batch_size):
            # Get the valid tokens for this example (non-padded and in response part)
            valid_mask = inputs["loss_attention_mask"][i].bool()
            
            if valid_mask.any():
                # Extract only the tokens we care about for the loss
                example_logits = logits[i, :-1][valid_mask[:-1]]  # Shift left: predict next token
                example_labels = inputs["input_ids"][i, 1:][valid_mask[:-1]]  # Shift right: actual next token
                
                if len(example_logits) > 0 and len(example_labels) > 0:
                    # Calculate cross-entropy loss for this example
                    example_loss = torch.nn.functional.cross_entropy(
                        example_logits.view(-1, example_logits.size(-1)),
                        example_labels.view(-1),
                        reduction='mean'
                    )
                    
                    losses.append(example_loss)
        
        # Combine losses from all examples
        if losses:
            loss = torch.stack(losses).mean()
        else:
            # Fallback to model's loss if we couldn't calculate our own
            loss = outputs.loss
    else:
        # Use the model's default loss calculation
        loss = outputs.loss
    
    # Split good and bad examples for contrastive loss
    if loss.dim() > 0 and loss.shape[0] == batch_size:
        # If the loss is already per-example
        good_loss = loss[is_good == 1].mean() if (is_good == 1).any() else torch.tensor(0.0, device=loss.device)
        bad_loss = loss[is_good == 0].mean() if (is_good == 0).any() else torch.tensor(0.0, device=loss.device)
    else:
        # Recalculate loss separately for good and bad examples
        good_indices = (is_good == 1).nonzero(as_tuple=True)[0]
        bad_indices = (is_good == 0).nonzero(as_tuple=True)[0]
        
        if len(good_indices) > 0:
            good_inputs = {k: v[good_indices] for k, v in inputs.items() if isinstance(v, torch.Tensor)}
            good_outputs = model(**good_inputs)
            good_loss = good_outputs.loss
        else:
            good_loss = torch.tensor(0.0, device=loss.device)
            
        if len(bad_indices) > 0:
            bad_inputs = {k: v[bad_indices] for k, v in inputs.items() if isinstance(v, torch.Tensor)}
            bad_outputs = model(**bad_inputs)
            bad_loss = bad_outputs.loss
        else:
            bad_loss = torch.tensor(0.0, device=loss.device)
    
    # Contrastive loss: minimize good loss, maximize bad loss
    contrastive_loss = good_loss - bad_loss
    
    if return_outputs:
        return contrastive_loss, outputs
    return contrastive_loss


class ContrastiveTrainer(SFTTrainer):
    """Trainer that uses contrastive loss for SFT training."""
    
    def __init__(self, *args, train_on_responses_only=False, instruction_part=None, 
                response_part=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_on_responses_only = train_on_responses_only
        self.instruction_part = instruction_part
        self.response_part = response_part
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Calculate the contrastive loss.
        
        Args:
            model: The model to compute loss for
            inputs: The inputs to the model
            return_outputs: Whether to return model outputs
            num_items_in_batch: Number of items in the batch (used by SFTTrainer)
            
        Returns:
            The contrastive loss, or a tuple of (loss, outputs) if return_outputs=True
        """
        return custom_contrastive_loss(
            model, 
            inputs, 
            return_outputs=return_outputs,
            train_on_responses_only=self.train_on_responses_only,
            instruction_part=self.instruction_part,
            response_part=self.response_part
        ) 