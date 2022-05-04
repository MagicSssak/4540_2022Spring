import torch
import torch.nn as nn
import torch.nn.functional as F



class FFN(nn.Module):
    def __init__(self, config):
        super(FFN, self).__init__()
        self.fc1 = nn.Linear(config.hidden_state, config.transformer['mlp_dim'])#wm,786->3072
        self.fc2 = nn.Linear(config.transformer['mlp_dim'], config.hidden_state)#wm,3072->786
        self.act_fn = F.gelu#wm,act func
        self.dropout = nn.Dropout(config.transformer['dropout'])

        self._init_weights(config.init_type)

    def _init_weights(self, init_type = 'normal'):
        if init_type == 'normal':
            nn.init.xavier_normal_(self.fc1.weight)
            nn.init.xavier_normal_(self.fc2.weight)
        else:
            nn.init.xavier_uniform_(self.fc1.weight)
            nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)#wm,786
        x = self.act_fn(x)
        x = self.dropout(x)#wm
        x = self.fc2(x)#wm

        return self.dropout(x)
