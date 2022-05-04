import torch
import torch.nn as nn
import math
import copy
from FFN import FFN
from Attention import Attention

class Block_E(nn.Module):

    def __init__(self, config):
        super(Block_E, self).__init__()
        self.dropout = nn.Dropout(p=config.dropout)
        self.hidden_state = config.hidden_state
        self.att_norm = nn.LayerNorm(self.hidden_state, eps = 1e-5)
        self.att = Attention(config)
        self.ffn_norm = nn.LayerNorm(self.hidden_state, eps = 1e-5)
        self.ffn = FFN(config)


    def forward(self, x, memory = None):

        r = x
        x = self.att_norm(x)
        x, weights = self.att(x)
        x = x + r

        r = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + r
        return x, weights



class Block_D(nn.Module):

    def __init__(self, config):

        super(Block_D, self).__init__()
        self.dropout_rate = config.transformer['dropout']
        self.hidden_state = config.hidden_state
        self.att_norm = nn.LayerNorm(self.hidden_state, eps = 1e-5)
        self.self_attn = Attention(config)
        self.multi_attn = Attention(config)
        self.ffn_norm = nn.LayerNorm(self.hidden_state, eps = 1e-5)
        self.ffn = FFN(config)
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, x, memory = None):

        r = x
        x = self.att_norm(x)
        x, _ = self.self_attn(x)
        x = r + x

        r = x
        x = self.att_norm(x)
        x, weights = self.multi_attn(x, memory)
        memory = None
        x = r + x

        r = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = r + x

        return self.dropout(x), weights

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class blocks(nn.Module):

    def __init__(self, mod, n, norm = None):
        super(blocks, self).__init__()
        self.layers = _get_clones(mod,n)
        self.norm = norm

    def forward(self, x, memory = None):

        for mod in self.layers:
            x, weights = mod(x, memory)
            memory = None

        if self.norm is not None:
            x = self.norm(x)

        return x, weights