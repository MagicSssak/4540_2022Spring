import torch
import torch.nn as nn
import ml_collections
import math

from Embeddings import Sinusoidal_embedding, Learnable_embedding
from config import get_config

config = get_config()
class Embeddings(nn.Module):
    '''
    对图像进行编码，把图片当做一个句子，把图片分割成块，每一块表示一个单词
    '''

    def __init__(self, config,  in_channels=3, pe_type = 'learnable'):
        super(Embeddings, self).__init__()
        self.img_size = config.img_size  # 224
        patch_size = config.patches["size"]  # 16
        ## （224/16）*（224/16）=196
        n_patches = (self.img_size // patch_size) * (self.img_size // patch_size)
        # （768）
        self.patch_embeddings = nn.Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_state,
                                       kernel_size=patch_size,
                                       stride=patch_size)

        self.pe_type = pe_type

        # （1,196+1,786）
        if pe_type == 'sinu':
            self.embedding = Sinusoidal_embedding(config)
        else:
            self.embedding = Learnable_embedding(config,n_patches)

        # 设置可学习的分类信息的维度
        self.classifer_token = nn.Parameter(torch.zeros(1, 1, config.hidden_state))
        self.dropout = nn.Dropout((config.transformer['dropout']))

    def forward(self, x):
        bs = x.shape[0]
        cls_tokens = self.classifer_token.expand(bs, -1, -1) # (bs, 1, 768)
        x = self.patch_embeddings(x)  # （bs,768,14,14）
        x = x.flatten(2)  # (bs,768,196)
        x = x.transpose(-1, -2)  # (bs,196,768)
        x = torch.cat((cls_tokens, x), dim=1)
        embeddings = self.embedding(x)
        return embeddings


