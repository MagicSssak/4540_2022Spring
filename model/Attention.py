import torch
import torch.nn as nn
import math



class Attention(nn.Module):
    def __init__(self, config, return_weights = False):
        super(Attention,self).__init__()
        self.device = config.device
        self.mask_type = config.mask_type
        self.return_weights = return_weights
        self.num_attention_heads=config.transformer["num_heads"]#12
        self.attention_head_size = int(config.hidden_state / self.num_attention_heads)  # 768/12=64
        self.all_head_size = self.num_attention_heads * self.attention_head_size  # 12*64=768

        self.query = nn.Linear(config.hidden_state, self.all_head_size)#wm,768->768，Wq矩阵为（768,768）
        self.key = nn.Linear(config.hidden_state, self.all_head_size)#wm,768->768,Wk矩阵为（768,768）
        self.value = nn.Linear(config.hidden_state, self.all_head_size)#wm,768->768,Wv矩阵为（768,768）
        self.out = nn.Linear(config.hidden_state, config.hidden_state)  # wm,768->768
        self.attn_dropout = nn.Dropout(config.dropout)
        self.proj_dropout = nn.Dropout(config.dropout)

        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)  # wm,(bs,197)+(12,64)=(bs,197,12,64)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)  # wm,(bs,12,197,64)

    def forward(self, x, memory = None):
        # hidden_states为：(bs,197,768)
        if memory is None:
            mixed_query_layer = self.query(x)#wm,768->768
            mixed_key_layer = self.key(x)#wm,768->768
            mixed_value_layer = self.value(x)#wm,768->768
        else:
            mixed_query_layer = self.query(memory)  # wm,768->768
            mixed_key_layer = self.key(memory)  # wm,768->768
            mixed_value_layer = self.value(x)  # wm,768->768


        # wm，(bs,12,197,64)
        q = self.transpose_for_scores(mixed_query_layer)
        k = self.transpose_for_scores(mixed_key_layer)
        v = self.transpose_for_scores(mixed_value_layer)
        mask_dim = x.size()[1]
        if self.mask_type == 'softmax':
            mask = self._generate_square_subsequent_mask(mask_dim).to(self.device)
            # （bs,12,197,197)
            att_score = torch.matmul(q, k.transpose(-1, -2))#（bs,12,197,197)

            att_score = att_score / math.sqrt(self.attention_head_size)#将结果除以向量维数的开方
            att_score = att_score + mask
            att_prob = self.softmax(att_score)#将得到的分数进行softmax,得到概率



        elif self.mask_type == 'gaussian':
            mask = self._generate_square_subsequent_mask(mask_dim).to(self.device)
            dif = q-k
            att_score = torch.exp(-torch.matmul(dif, dif.transpose(-1,-2))/2)
            att_prob = torch.mul(att_score, mask)

        else:
            att_score = torch.matmul(q, k.transpose(-1,-2))

            att_score = att_score / math.sqrt(self.attention_head_size)
            att_prob = self.softmax(att_score)


        weights = att_prob if self.return_weights else None  # wm
        att_prob = self.attn_dropout(att_prob)

        context_layer = torch.matmul(att_prob, v)#将概率与内容向量相乘

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)#wm,(bs,197)+(768,)=(bs,197,768)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)

        # wm,(bs,197,768),(bs,197,197)
        return attention_output, weights

    def _generate_square_subsequent_mask(self, sz):

        if self.mask_type == 'softmax':
            mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

        else:
            mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
            mask = mask.float().masked_fill(mask == 0, float(1)).masked_fill(mask == 1, float(0.0))
        return mask


