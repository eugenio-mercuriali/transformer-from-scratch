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

    # Add and norm - layer normalization
    # Gamma (alpha, multiplicative) and Beta(bias, additive) are parameters that we apply to each item
    # on top of the layer normalization and that the model can also learn, so the model can have the
    # possibility to amplify these values when he needs to
    class LayerNormalization(nn.Module):
        # We need epsilon since our CPU or GPU can only represent numbers up to a certain position
        # and also to avoid division by 0
        def __init__(self, eps: float = 10**-6) -> None:
            super().__init__()
            self.eps = eps
            # nn.Parameter to make them learnable
            self.alpha = nn.Parameter(torch.ones(1))  # multiplicative
            self.bias = nn.Parameter(torch.zeros(1))  # additive

        def forward(self, x):
            mean = x.mean(dim=-1, keepdim=True)
            std = x.std(dim=-1, keepdim=True)

            return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)  # W1 and B1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_model, d_ff)  # W2 and B2

    def forward(self, x):
        # Input sentence
        # (Batch, seq_len, d_model) --> (Batch, seq_len, d_ff) --> (Batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


class MultiHeadAttentionBlock(nn.Module):
    # h is the number of heads
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h

        # d_model should be divisible by h, otherwise we cannot divide equally the same vector
        # representing the embedding into equal matrices for each head
        assert d_model % h == 0, 'd_model is not divisible by h'

        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model)  # Wq
        self.w_k = nn.Linear(d_model, d_model)  # Wk
        self.w_v = nn.Linear(d_model, d_model)  # Wv

        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.attention_scores = None

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]  # taking the last dimension

        # @ is the matrix multiplication in PyTorch
        # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            # Replace all the values for which mask == 0 with a very small value
            # When we apply the softmax, these values will be replaced by 0s
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1)  # (batch, h, seq_len, seq_len)
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        # We return the attention scores for visualisation purposes
        return (attention_scores @ value), attention_scores

    def forward(self, x, q, k, v, mask):
        query = self.w_q(q)  # shape: (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        key = self.w_k(k)  # shape: (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        value = self.w_v(v)  # shape: (batch, seq_len, d_model) --> (batch, seq_len, d_model)

        # We are not splitting the batch or the sequence length, but the embedding
        # We transpose since we want the h dimension to be the second dimension instead of the third
        # In this way each head will see all the sentence (sequence length and d_k)
        # Each head will watch each word of the sentence but a smaller part of the embedding

        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.d_k).transpose(1, 2)

        # We do the same for the key and the value
        key = key.view(key.shape[0], key.shape[1], self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.d_k).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # (batch, h, seq_len, d_k) --> (batch, h, seq_len, d_k) --> (batch, seq_len, d_model)
        # self.h * self.d_k is equal to d_model
        x = x.transpose(1, 2).contiguos().view(x.shape[0], -1, self.h * self.d_k)

        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        return self.w_o(x)






