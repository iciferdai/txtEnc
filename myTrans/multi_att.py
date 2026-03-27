import logging

from myTrans.base_params import *

def dot_att(q, k, v, mask=None):
    att_scores = torch.matmul(q, k.transpose(-2, -1))
    att_scores /= np.sqrt(D_K)
    if mask is not None:
        att_scores = att_scores.masked_fill(mask, -1e9)
    att_weight = F.softmax(att_scores, dim=-1)
    att_weight = F.dropout(att_weight, p=DROPOUT_RATE)
    o = torch.matmul(att_weight, v)
    return o, att_weight

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.w_q = nn.Linear(D_MODEL, D_MODEL)
        self.w_k = nn.Linear(D_MODEL, D_MODEL)
        self.w_v = nn.Linear(D_MODEL, D_MODEL)
        self.w_o = nn.Linear(D_MODEL, D_MODEL)

    def split_heads(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1, NUM_HEADS, D_K)
        x = x.transpose(1, 2)
        return x

    def combine_heads(self, x):
        batch_size = x.shape[0]
        x = x.transpose(1, 2)
        x = x.contiguous().view(batch_size, -1, D_MODEL)
        return x

    def forward(self, q, k, v, mask=None):
        # 1. Linear
        q = self.w_q(q)
        k = self.w_k(k)
        v = self.w_v(v)

        # 2
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        # 3. attention
        att_o, att_w = dot_att(q=q, k=k, v=v, mask=mask)
        if not self.training:
            logging.debug(f'dot_att in: {q.shape}|{k.shape}|{v.shape}, out: {att_o.shape}|{att_w.shape}')

        # 4
        o = self.combine_heads(att_o)

        # 5. linear
        o = self.w_o(o)

        return o, att_w