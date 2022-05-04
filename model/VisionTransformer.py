import torch.nn as nn
from Transformer import Transformer


class VisionTransformer(nn.Module):

    def __init__(self, config, num_classes, zero_head=False):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.hidden_state = config.hidden_state

        self.transformer = Transformer(config)
        self.head = nn.Linear(self.hidden_state, self.num_classes)#wm,768-->10

    def forward(self, x, labels=None):
        x, attn_weights = self.transformer(x)
        #print(x)
        logits = self.head(x[:, 0])


        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
            return loss
        else:
            return logits, attn_weights

