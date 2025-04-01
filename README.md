# LLM Fine-tuning with LoRA

This project provides a framework for fine-tuning large language models using LoRA (Low-Rank Adaptation) on a single GPU. It includes utilities for training, testing, and uploading models to HuggingFace Hub.

## Project Structure

```
finetune_diffing/
├── config.py              # Configuration parameters
├── train.py              # Main training script
├── test_model.py         # Script for testing the trained model
├── requirements.txt      # Project dependencies
├── data/                 # Data directory
│   └── generate_dummy_data.py  # Script to generate dummy training data
└── outputs/              # Directory for model outputs (created during training)
```

## Setup

1. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your HuggingFace token (required for model upload):
```bash
export HUGGINGFACE_TOKEN=your_token_here
```

## Usage

1. Generate dummy training data (for development):
```bash
python data/generate_dummy_data.py
```

2. Configure your training parameters in `config.py`:
   - Set your base model
   - Configure LoRA parameters
   - Set your HuggingFace model ID for uploading

3. Train the model:
```bash
python train.py
```

4. Test the trained model:
```bash
python test_model.py
```

## Configuration

The `config.py` file contains all configurable parameters:
- Model configuration (base model, output directory)
- LoRA parameters (rank, alpha, dropout)
- Training parameters (batch size, learning rate, epochs)
- Dataset configuration
- Generation parameters for testing

## Notes

- The project uses LoRA for efficient fine-tuning on a single GPU
- Training progress is logged to Weights & Biases
- Only LoRA adapters are uploaded to HuggingFace Hub
- The base model should be accessible from HuggingFace Hub