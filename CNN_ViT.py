import torch
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
# import sparselinear as sl
from helpers import kmeans_clustering,topk_operation
# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes
class CNNPatchEmbedding(nn.Module):
    def __init__(self, image_height, image_width, patch_dim, in_channels, dim):
        super(CNNPatchEmbedding, self).__init__()
        self.dim = dim
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.max_pooling = nn.MaxPool2d(kernel_size=3,stride=1,padding=1)
        self.conv3 = nn.Conv2d(32, dim, kernel_size=(patch_dim, patch_dim), stride=(patch_dim, patch_dim))


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x= self.max_pooling(x)
        patches = self.conv3(x)
        patches = Rearrange('b c h w -> b (h w) c')(patches)
        return patches

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class CNNFeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=dim+1, out_channels=hidden_dim, kernel_size=1, stride=1),
            nn.MaxPool1d(kernel_size=1, stride=1),
            nn.Dropout(dropout),
            nn.GELU(), 
            nn.Conv1d(in_channels=hidden_dim, out_channels=dim+1, kernel_size=1),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.net(x)

        return x
    


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            # sl.SparseLinear(inner_dim, dim),
            nn.Linear(inner_dim,dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)') 

        out= self.to_out(out)
        return out,attn
    
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                # PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                # PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
                # PreNorm(dim,CNNFeedForward(dim,mlp_dim,dropout=dropout))
            ]))
    def forward(self, x):
        intermediate_feautres=[]
        attn_matrices = []
        for attn, ff in self.layers:
            x_attn,attn_matrix = attn(x)
            x+=x_attn
            x = ff(x) + x
            attn_matrices.append(attn_matrix)
            intermediate_feautres.append(x)
        return x,attn_matrices

    

class ViTCNNFF(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding_conv2d = CNNPatchEmbedding(image_height,image_width,patch_size,channels,dim) # patch embedding + convoluciones

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):

        x = self.to_patch_embedding_conv2d(img) # camiar a self.patch_embedding_linear para el original
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        #####
        intermediate_features = []
        
        # if self.distilling:
        #     distill_tokens = repeat(self.distill_token, '() n d -> b n d', b = b)
        #     x = torch.cat((x, distill_tokens), dim = 1)

        x, attn_matrices = self.transformer(x)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)

        x = self.mlp_head(x)

        # if self.distilling:
        #     distill_token = distill_tokens[:, 0]
        #     distill_token = self.to_latent(distill_tokens)
        #     distill_logits = self.mlp_head(distill_token)
        #     return x[:, :-1], distill_logits[:,:-1] 
        
        return x, attn_matrices
        ########
    
    



class DynamicGroupAttention(nn.Module):
    def __init__(self, dim, num_clusters, tau, lr, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        self.num_clusters = num_clusters
        self.tau = tau
        self.lr = lr
        self.attention = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.k = 5  
        
    def forward(self, x):
        B, N, _ = x.size()
        X_query = x[:, 1:]  
        cluster_indices = kmeans_clustering(X_query, self.num_clusters)
        XK, XV = topk_operation(X_query, cluster_indices, self.num_clusters, self.k)
        Y = self.attention(X_query, XK, XV)
        return torch.cat((x[:, :1], Y), dim=1)  
    

class ViTWithDynamicGroupAttention(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, num_clusters, tau, lr, pool='cls', channels=3, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        transformer_layers = []
        for _ in range(depth):
            transformer_layers.extend([
                DynamicGroupAttention(dim, num_clusters, tau, lr),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ])

        self.transformer = nn.ModuleList(transformer_layers)

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

        for layer in self.transformer:
            x = layer(x) + x

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)
