from torch import nn
import torch.nn.functional as F
import torch
import copy
import math



class ScaledDotProductAttention(nn.Module):

    def forward(self, query, key, value, mask=None):
        dk = query.size()[-1]
        scores = query.matmul(key.transpose(-2, -1)) / math.sqrt(dk)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = F.softmax(scores, dim=-1)
        return attention.matmul(value)


class MultiHeadAttention_another(nn.Module):

    def __init__(self,
                 in_features,
                 head_num,
                 bias=True,
                 activation=F.relu):
        """Multi-head attention.
        :param in_features: Size of each input sample.
        :param head_num: Number of heads.
        :param bias: Whether to use the bias term.
        :param activation: The activation after each linear transformation.
        """
        super(MultiHeadAttention_another, self).__init__()
        if in_features % head_num != 0:
            raise ValueError('`in_features`({}) should be divisible by `head_num`({})'.format(in_features, head_num))
        self.in_features = in_features
        self.head_num = head_num
        self.activation = activation
        self.bias = bias
        self.linear_q = nn.Linear(in_features, in_features, bias)
        self.linear_k = nn.Linear(in_features, in_features, bias)
        self.linear_v = nn.Linear(in_features, in_features, bias)
        self.linear_o = nn.Linear(in_features, in_features, bias)
        self.gathering = nn.Linear(10,1)

    def forward(self, q, k, v, mask=None):
        q, k, v = self.linear_q(q), self.linear_k(k), self.linear_v(v)
        if self.activation is not None:
            q = self.activation(q)
            k = self.activation(k)
            v = self.activation(v)

        q = self._reshape_to_batches(q)
        k = self._reshape_to_batches(k)
        v = self._reshape_to_batches(v)
        if mask is not None:
            mask = mask.repeat(self.head_num, 1, 1)
        y = ScaledDotProductAttention()(q, k, v, mask)
        y = self._reshape_from_batches(y)

        y = self.linear_o(y)
        if self.activation is not None:
            y = self.activation(y)
        y = y.permute(0,2,1)
        y = self.gathering(y)
        y = y.permute(0,2,1).squeeze()

        return y

    @staticmethod
    def gen_history_mask(x):
        """Generate the mask that only uses history data.
        :param x: Input tensor.
        :return: The mask.
        """
        batch_size, seq_len, _ = x.size()
        return torch.tril(torch.ones(seq_len, seq_len)).view(1, seq_len, seq_len).repeat(batch_size, 1, 1)

    def _reshape_to_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        sub_dim = in_feature // self.head_num
        return x.reshape(batch_size, seq_len, self.head_num, sub_dim)\
                .permute(0, 2, 1, 3)\
                .reshape(batch_size * self.head_num, seq_len, sub_dim)

    def _reshape_from_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        batch_size //= self.head_num
        out_dim = in_feature * self.head_num
        return x.reshape(batch_size, self.head_num, seq_len, in_feature)\
                .permute(0, 2, 1, 3)\
                .reshape(batch_size, seq_len, out_dim)

    def extra_repr(self):
        return 'in_features={}, head_num={}, bias={}, activation={}'.format(
            self.in_features, self.head_num, self.bias, self.activation,
        )

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class MutilheadAttention(nn.Module):
    # n_heads：多头注意力的数量
    # hid_dim：每个词输出的向量维度
    def __init__(self, hid_dim, n_heads, dropout):
        super(MutilheadAttention, self).__init__()
        self.hid_dim = hid_dim
        self.n_heads = n_heads

        assert hid_dim % n_heads == 0
        # 定义 W_q 矩阵
        self.w_q = nn.Conv2d(hid_dim,hid_dim,kernel_size=1)
        # 定义 W_k 矩阵
        #self.w_k = nn.Linear(hid_dim,hid_dim)
        self.w_k = nn.Conv2d(hid_dim,hid_dim,kernel_size=1)
        # 定义 W_v 矩阵
        self.w_v = nn.Conv2d(hid_dim,hid_dim,kernel_size=1)
        #self.fc = nn.Conv2d(hid_dim,hid_dim,kernel_size=1)
        self.do = nn.Dropout(dropout)
        # 缩放
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads]))

    def forward(self, query, key, value, mask=None):
        bsz = query.shape[0]
        b,c, h, w = query.shape
        temp=query.to('cuda')
        Q = self.w_q(query).to('cuda')
        # K = self.w_k(key)
        K = self.w_k(key).to('cuda')
        V = self.w_v(value).to('cuda')
        V = V.permute(0, 2, 3, 1)
        V = torch.flatten(V, 1, 2)
        Q = Q.permute(0, 2, 3, 1)
        Q = torch.flatten(Q, 1, 2)
        K = K.permute(0, 2, 3, 1)
        K = torch.flatten(K, 1, 2)
        # 转置是为了把注意力的数量 放到前面，方便下面计算
        Q = Q.view(bsz, -1, self.n_heads, self.hid_dim //
                   self.n_heads).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.n_heads, self.hid_dim //
                   self.n_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.n_heads, self.hid_dim //
                   self.n_heads).permute(0, 2, 1, 3)
        attention = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale.to('cuda')
        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e10)
        attention = self.do(torch.softmax(attention, dim=-1))
        x = torch.matmul(attention,V)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(bsz, -1, self.n_heads * (self.hid_dim // self.n_heads))
        x = x.permute(0, 2, 1)
        x = x.reshape(b,c, h, w)
        x=x+temp
        return x


query = torch.rand(80, 640,5,5)
key = torch.rand(80, 640,5,5)
value = torch.rand(80,640,5,5)
attention = MutilheadAttention(hid_dim=640, n_heads=8, dropout=0.1)
output = attention(query, key, value)
#print(output.shape)

