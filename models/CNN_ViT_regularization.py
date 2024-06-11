import torch
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, num_classes, early_exit, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.depth = depth
        self.early_exit = early_exit

        # Convolutional layer
        self.conv = nn.Conv2d(dim, dim, kernel_size=2, stride=1, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(dim, num_classes)

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))
        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, x, confidence_threshold=0.9):
        early_exit_info = {'exited': False, 'confidence': 0.0, 'correct': False}
        cnn_logits = None

        for idx, (attn, ff) in enumerate(self.layers):
            x_attn = attn(x)
            x = x + x_attn
            x = x + ff(x)

            # Always compute CNN output for backpropagation purposes
            if idx == self.depth - 2:
                x = rearrange(x, 'b n d -> b d n 1')
                x = self.conv(x)
                x = self.pool(x)
                cnn_out = x
                x = rearrange(x, 'b d n 1 -> b n d')
                cnn_logits = rearrange(cnn_out, 'b d 1 1 -> b d')
                cnn_logits = self.fc(cnn_logits)
                
                # x = x + cnn_out
                
                if self.early_exit:
                    probabilities = torch.softmax(cnn_logits, dim=-1)
                    max_probability, predicted = torch.max(probabilities, dim=-1)
                    confidence = max_probability.item()
                    
                    print(confidence)
                    if confidence > confidence_threshold:
                        early_exit_info['exited'] = True
                        early_exit_info['confidence'] = confidence
                        return cnn_logits, early_exit_info
        x = x.mean(dim=1)
        final_logits = self.mlp_head(x)
        return final_logits if cnn_logits is None else (final_logits + cnn_logits), early_exit_info
        # x = x[:, 0]
        # # Agrega la salida de la CNN a la salida cls_token
        # x += early_exit_logits.squeeze(1) if early_exit_logits is not None else 0

        # x = self.mlp_head(x)
        # return x, early_exit_info

class CNN_ViT_early_exit(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=3, dim_head=64, dropout=0., emb_dropout=0., early_exit=False):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.early_exit = early_exit

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, num_classes, early_exit, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        logits,early_exit_info = self.transformer(x)
        # logits = self.transformer(x)
        return logits, early_exit_info