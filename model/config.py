import ml_collections
import torch


def get_config():
    """Returns the ViT-B/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size':8})

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
    config.init_type = 'normal'
    config.num_classes = 10
    config.img_size = 224
    config.eval_batch_size = 16
    config.train_batch_size = 16
    config.total_epoch = 1000

    config.dataset = 'cifar10'
    config.train_dir = r'C:\Users\SssaK\Desktop\4540\train'
    config.test_dir = r'C:\Users\SssaK\Desktop\4540\test'
    config.output_dir = r'C:\Users\SssaK\Desktop\4540\output'

    config.learning_rate = 0.01
    config.weight_decay = 0.01

    config.device = torch.device(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
    return config
