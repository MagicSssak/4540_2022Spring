import torch
import torch.nn as nn
import copy
import torch.nn.functional as F
import math

from config import get_config
from embedding import Embeddings
from Blocks import *



class Transformer(nn.Module):

    def __init__(self, config):
        super(Transformer, self).__init__()
        self.num_layers = config.transformer['num_layers']
        self.embeddings = Embeddings(config)
        self.encoder_block = Block_E(config)
        self.decoder_block = Block_D(config)
        self.encoder_layer = blocks(self.encoder_block, self.num_layers)
        self.decoder_layer = blocks(self.decoder_block, self.num_layers)
        self.dropout = nn.Dropout(config.transformer['dropout'])


    def forward(self, x):

        embedded = self.embeddings(x)
        encoded, _ = self.encoder_layer(embedded)

        output, weights = self.decoder_layer(embedded, encoded)

        return self.dropout(output), weights


