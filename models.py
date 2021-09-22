from swin_transformer import SwinTransformer
from vit import ViT
from functools import partial
import torch.nn as nn
import torch


class ViT_Breg(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, breg_dim=128, hidden_size=128, 
                 d_subs=200, bn=True):
        super().__init__()
        
        self.d_subs = d_subs
        
        self.h = img_size // patch_size
        self.dim = embed_dim
        
        
        
        # self.backbone = VisionTransformer(img_size=img_size, patch_size=patch_size,
        #                                 in_chans=in_chans, num_classes=num_classes,
        #                                 embed_dim=embed_dim, depth=depth,
        #                                 num_heads=num_heads, norm_layer=partial(nn.LayerNorm, eps=1e-6))
        
        self.backbone = ViT(image_size=img_size, patch_size=patch_size, num_classes=num_classes,
                            dim=embed_dim, depth=depth, heads=num_heads, channels=in_chans,
                            mlp_dim=4*embed_dim)
        
        
        
        self.build_2d_sincos_position_embedding()
        
        #Freeze Projection Layer for more stable training according to MoCo-v3
        self.backbone.to_patch_embedding.requires_grad_(False)
        #self.backbone.to_patch_embedding.linear.bias.requires_grad = False
        
        
        self.mlp_head = nn.Sequential(
            nn.BatchNorm1d(embed_dim), 
            nn.Linear(embed_dim, 4096),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(4096),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(4096),
            nn.Linear(4096, breg_dim),
            nn.BatchNorm1d(breg_dim)
        )
        
        self.predictor = nn.Sequential(
            nn.Linear(breg_dim, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, breg_dim),
        )
        
        #Remove last fc layer from backbone
        del self.backbone.mlp_head
        self.backbone.mlp_head = self.mlp_head
        
        
        if bn==True:
           self.subs = nn.Sequential(
                    nn.Conv1d(in_channels=breg_dim*d_subs, out_channels=hidden_size*d_subs,
                              kernel_size=1, groups=d_subs),
                    nn.BatchNorm1d(hidden_size*d_subs),
                    nn.Conv1d(in_channels=hidden_size*d_subs, out_channels=d_subs, 
                              kernel_size=1, groups=d_subs))
        else:
           self.subs = nn.Sequential(
                    nn.Conv1d(in_channels=breg_dim*d_subs, out_channels=hidden_size*d_subs,
                              kernel_size=1, groups=d_subs),
                    nn.Conv1d(in_channels=hidden_size*d_subs, out_channels=d_subs, 
                              kernel_size=1, groups=d_subs))
           
    def build_2d_sincos_position_embedding(self, temperature=10000.):
        '''From MoCov3 Code!!'''
        h, w = self.h, self.h
        grid_w = torch.arange(w, dtype=torch.float32)
        grid_h = torch.arange(h, dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h)
        assert self.dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = self.dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature**omega)
        out_w = torch.einsum('m,d->md', [grid_w.flatten(), omega])
        out_h = torch.einsum('m,d->md', [grid_h.flatten(), omega])
        pos_emb = torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], dim=1)[None, :, :]

        #assert self.num_tokens == 1, 'Assuming one and only one token, [cls]'
        pe_token = torch.zeros([1, 1, self.dim], dtype=torch.float32)
        self.backbone.pos_embedding = nn.Parameter(torch.cat([pe_token, pos_emb], dim=1))
        self.backbone.pos_embedding.requires_grad = False
           
    def forward(self, x):
        x = self.backbone.forward(x)
        x = self.predictor(x)
        
        sub_x = x.repeat(1, self.d_subs)
        out = self.subs(sub_x.unsqueeze(dim=2))
        return x, out.squeeze(dim=2)
    
    
    

class Swin_Breg(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24),
                 breg_dim=128, hidden_size=128, d_subs=200, mlp_dim=768, norm_before_nlp='bn', bn=True):
        super().__init__()
        
        self.d_subs = d_subs

        self.backbone = SwinTransformer(img_size=img_size, patch_size=patch_size,
                                        in_chans=in_chans, num_classes=num_classes,
                                        embed_dim=embed_dim, depths=depths,
                                        num_heads=num_heads)
        
        self.mlp_head = nn.Sequential(
            nn.BatchNorm1d(mlp_dim), 
            nn.Linear(mlp_dim, 4096),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(4096),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(4096),
            nn.Linear(4096, breg_dim),
            nn.BatchNorm1d(breg_dim)
        )
        
        self.predictor = nn.Sequential(
            nn.Linear(breg_dim, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, breg_dim),
        )
        
        if bn==True:
           self.subs = nn.Sequential(
                    nn.Conv1d(in_channels=breg_dim*d_subs, out_channels=hidden_size*d_subs,
                              kernel_size=1, groups=d_subs),
                    nn.BatchNorm1d(hidden_size*d_subs),
                    nn.Conv1d(in_channels=hidden_size*d_subs, out_channels=d_subs, 
                              kernel_size=1, groups=d_subs))
        else:
           self.subs = nn.Sequential(
                    nn.Conv1d(in_channels=breg_dim*d_subs, out_channels=hidden_size*d_subs,
                              kernel_size=1, groups=d_subs),
                    nn.Conv1d(in_channels=hidden_size*d_subs, out_channels=d_subs, 
                              kernel_size=1, groups=d_subs))
           
         #Remove last fc layer from backbone
        del self.backbone.head
        self.backbone.head = self.mlp_head

        
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.predictor(x)
        
        sub_x = x.repeat(1, self.d_subs)
        out = self.subs(sub_x.unsqueeze(dim=2))
        return x, out.squeeze(dim=2)
    
    
