#https://github.com/kyegomez/SparseAttention
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

def get_attn_mask(n, attn_mode, local_attn_ctx=None):
    if attn_mode == 'all':
        b = torch.tril(torch.ones([n, n]))
    elif attn_mode == 'local':
        bandwidth = local_attn_ctx
        ctx = min(n - 1, bandwidth - 1)
        b = torch.tril(torch.ones([n, n]), ctx)
    elif attn_mode == 'strided':
        stride = local_attn_ctx
        x = torch.reshape(torch.arange(n, dtype=torch.int32), [n, 1])
        y = torch.transpose(x, 0, 1)
        z = torch.zeros([n, n], dtype=torch.int32)
        q = z + x
        k = z + y
        c1 = q >= k
        c2 = torch.eq(torch.fmod(q - k, stride), 0)
        c3 = torch.logical_and(c1, c2)
        b = c3.float()
    elif attn_mode == 'cross':
        center = n // 2
        start_idx = max(center - local_attn_ctx, 0)
        end_idx = min(center + local_attn_ctx + 1, n)
        
        # Create the mask tensor with ones in the specified range
        b = torch.zeros((n, n), dtype=torch.float32)
        b[start_idx:end_idx, :] = 1   
    else:
        raise ValueError('Not yet implemented')
    b = torch.reshape(b, [1, 1, n, n])
    return b

def strided_transpose(x, n_ctx, local_attn_ctx, blocksize):
    bT_ctx = n_ctx // local_attn_ctx
    assert bT_ctx % blocksize == 0, f'{bT_ctx}, {blocksize}'
    n, t, embd = x.size()
    x = torch.reshape(x, [n, bT_ctx, local_attn_ctx, embd])
    x = x.permute(0, 2, 1, 3).contiguous() 
    x = torch.reshape(x, [n, t, embd])
    return x

def split_heads(x, n):
    return split_states(x, n).permute(0, 2, 1, 3)




def merge_heads(x):
    return merge_states(torch.transpose(x, 1, 2)) #modificado


def split_states(x, n):
    """
    reshape (batch, pixel, state) -> (batch, pixel, head, head_state)
    """
    x_shape = x.size()
    m = x_shape[-1]
    new_x_shape = tuple(x_shape[:-1]) + (n, m // n)  
    return torch.reshape(x, new_x_shape)


    return torch.reshape(x, new_x_shape)

def merge_states(x):
    """
    reshape (batch, pixel, head, head_state) -> (batch, pixel, state)
    """
    x_shape = x.size()
    # new_x_shape = x_shape[:-2] + [np.prod(x_shape[-2:])]
    new_x_shape = x_shape[:-2] + (np.prod(x_shape[-2:]),) #modificado

    return torch.reshape(x, new_x_shape)

def attention_impl(q, k, v, heads, attn_mode, local_attn_ctx=None,dim=65):
    q = split_heads(q, heads)
    k = split_heads(k, heads)
    v = split_heads(v, heads)
    n_timesteps = k.size()[2]
    mask = get_attn_mask(n_timesteps, attn_mode, local_attn_ctx).float()
    w = torch.matmul(q, k.transpose(-2, -1))
    scale_amount = 1.0 / np.sqrt(q.size()[-1])
    w = w * scale_amount
    w=w.to('cuda')
    mask=mask.to('cuda')
    w = w * mask + -1e9 * (1 - mask)
    w = F.softmax(w, dim=-1)
    a = torch.matmul(w, v)
    a = merge_heads(a)
    return a,w

def blocksparse_attention_impl(q, k, v, heads, attn_mode, local_attn_ctx=None, blocksize=4, num_verts=None, vertsize=None):
    n_ctx = q.size()[1]

    # Handle the strided attention mode
    if attn_mode == 'strided':
        q = strided_transpose(q, n_ctx, local_attn_ctx, blocksize)
        k = strided_transpose(k, n_ctx, local_attn_ctx, blocksize)
        v = strided_transpose(v, n_ctx, local_attn_ctx, blocksize)

    n_state = q.size()[-1] // heads
    scale_amount = 1.0 / np.sqrt(n_state)

    # Attention weights
    w = torch.matmul(q, k.transpose(-2, -1))  # Ensure k is not modified in-place
    w=w.to('cpu')
    w = F.softmax(w * scale_amount, dim=-1)

    # Attention output
    a = torch.matmul(w, v)  # Ensure v is not modified in-place

    if attn_mode == 'strided':
        n, t, embd = a.size()
        bT_ctx = n_ctx // local_attn_ctx
        a = torch.reshape(a, [n, local_attn_ctx, bT_ctx, embd])
        a = a.permute(2, 1, 0, 3).contiguous()  # Swap dimensions without in-place modification
        a = a.permute(1, 0, 2, 3).contiguous()  # Swap dimensions without in-place modification
        a = torch.reshape(a, [n, t, embd])

    return a, w

class SparseAttention(nn.Module):
    def __init__(self, heads, attn_mode, local_attn_ctx=None, blocksize=4):
        super(SparseAttention, self).__init__()
        self.heads = heads
        self.attn_mode = attn_mode
        self.local_attn_ctx = local_attn_ctx
        self.blocksize = blocksize

    def forward(self, q, k, v):
        a,w= blocksparse_attention_impl(q, k, v, self.heads, self.attn_mode, self.local_attn_ctx,blocksize=4)
        # a,w=attention_impl(q, k, v, self.heads,self.attn_mode, self.local_attn_ctx,dim=65)
        return a,w


# Example usage:
if __name__ == "__main__":
    n_batch = 4
    n_ctx = 1024
    n_embd = 256
    heads = 4
    attn_mode = "all"
    local_attn_ctx = 32
    blocksize = 32


    q = torch.randn(n_batch, n_ctx, n_embd)
    k = torch.randn(n_batch, n_ctx, n_embd)
    v = torch.randn(n_batch, n_ctx, n_embd)

    model = SparseAttention(heads, attn_mode, local_attn_ctx, blocksize)
    output = model(q, k, v)
    print(output)
