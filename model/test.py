import torch
import torch.nn as nn
import ml_collections

from Attention import Attention
from embedding import Embeddings

from VisionTransformer import VisionTransformer

def get_config():
    """Returns the ViT-B/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size':16})

    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 128
    config.transformer.num_heads = 8
    config.transformer.num_layers = 8
    config.transformer.dropout = 0.1


    config.dropout = 0.0
    config.mask_type = 'softmax'
    config.hidden_state = 128
    config.classifier = 'token'
    config.representation_size = None
    config.init_type = 'uniform'
    config.num_classes = 10
    config.img_size = 224
    config.eval_batch_size = 128
    config.train_batch_size = 128
    config.total_epoch = 150

    config.dataset = 'cifar10'
    config.train_dir = r'C:\Users\SssaK\Desktop\4540\train'
    config.test_dir = r'C:\Users\SssaK\Desktop\4540\test'
    config.output_dir = r'C:\Users\SssaK\Desktop\4540\output'

    config.learning_rate = 0.01
    config.weight_decay = 0.01

    config.device = torch.device('cpu')
    return config


config = get_config()
img = torch.rand(2,3,224,224).to(config.device)


ViT = VisionTransformer(config, config.num_classes)
out, _ = ViT(img)
print(out.shape)
'''
tensor([[-0.9252,  0.2785, -0.9747, -1.3759,  4.8659,  5.3873,  0.5834,  1.5013,
          2.8378,  0.2334],
        [ 1.7794, -0.8563,  3.1012,  3.2576, 10.2118,  0.8426,  0.7738,  4.0148,
         -3.3202, -0.9961]])
'''


a = torch.randn(4,4)
print(out)
print(torch.max(out))
print('----------------------')
print(out.argmax(1))
print('----------------------')
print(torch.argmax(out,dim  = 1))