# GPT Model Training

This repository contains the implementation and training script for a GPT (Generative Pre-trained Transformer) model. The code is written in Python and utilizes PyTorch for building and training the model.

## Files

- `train_gpt.py`: The main script that defines the GPT model, its components, and the training loop.
- `input.txt`: The text file containing the training data.

## Requirements

- Python 3.8+
- PyTorch 1.10+
- Transformers library from Hugging Face
- tiktoken library

## Installation

Clone the repository:
```sh
git clone <repository-url>
cd learn-gpt
```

Install the required packages:
```sh
pip install -r requirements.txt
```

## Usage

### Prepare your training data:
Place your training text data in a file named `input.txt` in the root directory.

### Train the model:
```sh
python train_gpt.py
```

## Model Components

- **CausalSelfAttention**: A vanilla multi-head masked self-attention layer with a projection at the end.
- **MLP**: A simple class for a Multi-Layer Perceptron (MLP) that combines a linear layer with a GELU activation function.
- **Block**: Implements a single block of the GPT model, with a layer norm, a multi-head self-attention mechanism, and a feed-forward network.
- **GPTConfig**: A dataclass to hold the configuration of the GPT model, including hyperparameters and other variables.
- **GPT**: The main GPT model class that implements the forward pass and weight initialization.
- **DataLoaderLite**: A simple data loader that loads the entire text file into memory and serves data in batches.

## Training

The training script initializes the model, loads the data, and trains the model using the AdamW optimizer and cross-entropy loss. The training loop runs for a specified number of steps, printing the loss at each step.

## Generating Text

After training, you can generate text by providing a prefix and using the trained model to generate sequences.

## License

This project is licensed under the Apache License. See the LICENSE file for details.

## Acknowledgments

- The implementation is inspired by the GPT-2 model from OpenAI.
- The code uses the transformers library from Hugging Face for loading pretrained models.

Feel free to contribute to this project by opening issues or submitting pull requests.