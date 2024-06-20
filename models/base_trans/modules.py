import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F

class Residual(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x, **kwargs):
        return self.module(x, **kwargs) + x
    
class PreNorm(nn.Module):
    def __init__(self, dim, module):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.module = module

    def forward(self, x, **kwargs):
        return self.module(self.norm(x), **kwargs)
    
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.FFnet = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),    # 待实验，dropout是否有激励作用！！！！
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.FFnet(x)
    
class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dropout = 0.):
        super().__init__()
        
        self.heads = heads
        self.scale = heads ** -0.5

        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x, mask=None):
        # print('-' * 80)
        # print('this is attention')
        b, n, _, h = *x.shape, self.heads   # [B, 17, 64], 8
        # print(x.shape, 'input')
        qkv = self.to_qkv(x)    # [B, 17, 64*3]
        # print(qkv.shape, 'qkv')
        q, k, v = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', qkv=3, h=h)    # [B, 8, 17, 8]
        # print(q.shape, 'q')
        # print(k.shape, 'k')
        # print(v.shape, 'v')
        
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale    # [B, 8, 17, 17]
        # print(dots.shape, 'dots')

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], f'{mask.shape[1]} != {dots.shape[1]}'
            mask = mask[:, None, :] * mask[:, :, None]
            dots = dots.masked_fill(~mask, float("-inf"))
            del mask
        
        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v) # [B, 8, 17, 8]
        # print(out.shape, 'out')
        out = rearrange(out, 'b h n d -> b n (h d)')    # [B, 17, 64]
        # print(out.shape, 'out')
        out = self.to_out(out)  # [B, 17, 64]
        # print(out.shape, 'out')
        # print('-' * 80)
        return out
