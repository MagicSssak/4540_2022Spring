import torch
import torch.nn as nn
import ml_collections
import math
from config import get_config

config = get_config()

class Sinusoidal_embedding(nn.Module):

    def __init__(self, config, dropout=0.1, max_len=5000):
        super(Sinusoidal_embedding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, config.hidden_state)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, config.hidden_state, 2).float() * (-math.log(10000.0) / config.hidden_state))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        embeddings = x + self.pe[:x.size(0), :]
        return self.dropout(embeddings)





class Learnable_embedding(nn.Module):

    def __init__(self, config, n_patches):
        super(Learnable_embedding, self).__init__()

        self.n_patches = n_patches
        # （1,196+1,786）
        self.position_embeddings = nn.Parameter(torch.zeros(1,
                                                            self.n_patches + 1,
                                                            config.hidden_state))
        #
        self.dropout = nn.Dropout((config.transformer["dropout"]))

    def forward(self, x):

        embeddings = x + self.position_embeddings  # (bs,197,768)


        return self.dropout(embeddings)

