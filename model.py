import math
import torch
import torch.nn as nn


# Input embeddings
class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # In the original paper the authors multiply the embeddings
        # by the root of the size of the model
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)

        # Simplified calculation using log-space (for numerical stability)
        # Create a vector of shape (seq_len) (tensor of shape (seq_len, 1))
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        # Denominator of the formula (tensor built in log space)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() + (-math.log(10000.0) / d_model))
        # Apply the sin to the even positions
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply the cos to the odd positions
        pe[:, 1::2] = torch.cos(position * div_term)
        # Add batch dimension - now the shape is (seq_len, d_model)
        # We will instead have a batch of sentences
        pe = pe.unsqueeze(0)  # (1, seq_len, d_model)

        # When we have a tensor that we want to keep inside the model not as a learned parameter
        # that will be saved when we save the file of the model, we should register it as a buffer
        # In this way the tensor will be saved in the file along with the state of the model

        self.register_buffer('pe', pe)

    def forward(self, x):
        # We need to add the positional encoding to every word inside the sentence
        # We tell the model that we don't want to learn the positional encoding with requires_grad(False)
        # Since the values will always be the same
        x = x + (self.pe[:, x.shape[1], :]).requires_grad(False)
        # Apply the dropout to reduce overfitting
        return self.dropout(x)







