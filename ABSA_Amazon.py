
import torch
from torch import nn
import torchtext
import numpy as np
from gensim.models import Word2Vec
from tqdm import tqdm
import random
import re

# Define your ClassifierAttention model class here
class ClassifierAttention(nn.Module):
    def __init__(self, vocab_size, emb_dim, padding_idx, hidden_size, n_layers, attention_heads, hidden_layer_units, dropout):
        super(ClassifierAttention, self).__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=emb_dim,
            padding_idx=padding_idx
        )

        self.rnn_1 = nn.LSTM(
            emb_dim,
            hidden_size,
            n_layers,
            bidirectional=False,
            batch_first=True,
        )
        self.attention = Attention(hidden_size, attention_heads)

        self.rnn_2 = nn.LSTM(
            hidden_size,
            hidden_size,
            n_layers,
            bidirectional=False,
            batch_first=True,
        )

        self.dropout = nn.Dropout(dropout)

        hidden_layer_units = [hidden_size, *hidden_layer_units]
        self.hidden_layers = nn.ModuleList([])
        for in_unit, out_unit in zip(hidden_layer_units[:-1], hidden_layer_units[1:]):
            self.hidden_layers.append(nn.Linear(in_unit, out_unit))
            self.hidden_layers.append(nn.ReLU())
            self.hidden_layers.append(self.dropout)
        self.hidden_layers.append(nn.Linear(hidden_layer_units[-1], 1))

        self.sigmoid = nn.Sigmoid()

        self.add_custom_embeddings()

    def add_custom_embeddings(self):
        self.embedding.weight.data.copy_(pretrained_embeddings)
        self.embedding.weight.requires_grad = False

    def forward(self, x):
        out = self.embedding(x)
        out, (hidden_state, cell_state) = self.rnn_1(out)
        out = self.attention(out)
        out = self.dropout(out)
        output, (hidden_state, cell_state) = self.rnn_2(out)
        out = hidden_state[-1]

        for layer in self.hidden_layers:
            out = layer(out)

        out = self.sigmoid(out)
        out = out.squeeze(-1)

        return out

# Function to load the model
def load_model():
    model = ClassifierAttention(VOCAB_SIZE, EMB_DIM, PADDING_IDX, LSTM_HIDDEN_SIZE, LSTM_N_LAYERS, ATTENTION_HEADS, HIDDEN_LAYER_UNITS, DROPOUT).to(DEVICE)
    model.load_state_dict(torch.load('model.pt'))
    return model

# Function to preprocess text
def preprocess_text(text):
    text = clean_text(text)
    return TEXT_PIPELINE(text)

# Custom Attention layer
def self_attention(Q, K, V):
    d = K.shape[-1]
    QK = Q @ K.transpose(-2, -1)
    QK_d = QK / (d ** 0.5)
    weights = torch.softmax(QK_d, axis=-1)
    outputs = weights @ V
    return outputs

class Attention(torch.nn.Module):
    def __init__(self, emb_dim, n_heads):
        super(Attention, self).__init__()

        self.emb_dim = emb_dim
        self.n_heads = n_heads

    def forward(self, X):

        batch_size, seq_len, emb_dim = X.size()
        n_heads = self.n_heads
        emb_dim_per_head = emb_dim // n_heads

        assert emb_dim == self.emb_dim
        assert emb_dim_per_head * n_heads == emb_dim

        X = X.transpose(1, 2)
        output = self_attention(X, X, X)
        output = output.transpose(1, 2)
        output = output.contiguous().view(batch_size, seq_len, emb_dim)

        return output

# Other required functions

# ...

# Constants
VOCAB_SIZE = 8000
EMB_DIM = 128
PADDING_IDX = 1
LSTM_HIDDEN_SIZE = 64
LSTM_N_LAYERS = 1
ATTENTION_HEADS = 2
HIDDEN_LAYER_UNITS = [64, 64]
DROPOUT = 0.4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained word embeddings
model = Word2Vec(
    vector_size=128,
    window=5,
    min_count=1
)

# Vocabulary
model.build_vocab_from_freq({i: VOCAB_SIZE - i + 1 for i in range(VOCAB_SIZE)})
for k, v in model.wv.key_to_index.items():
    assert k == v

# Training
model.train(
    [i[1] for i in train],
    total_examples=len(train),
    epochs=3
)

# Extracting Embeddings
pretrained_embeddings = model.wv.vectors
pretrained_embeddings = torch.tensor(pretrained_embeddings)