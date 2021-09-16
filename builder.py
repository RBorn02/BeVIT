

from models import ViT_Breg, Swin_Breg

class Builder():
    def __init__(self, base_model, img_size, breg_dim,
                 d_subs, hidden_size, bn, device):
        
       if base_model == "ViT16-S":
           self.model = ViT_Breg(img_size=img_size, 
                    breg_dim=breg_dim, 
                    d_subs=d_subs,
                    embed_dim=384,
                    hidden_size=hidden_size,
                    bn=bn).to(device)
        
        
       elif base_model == "ViT16-B":
             self.model = ViT_Breg(img_size=img_size,
                    breg_dim=breg_dim, 
                    d_subs=d_subs,
                    hidden_size=hidden_size,
                    bn=bn).to(device)
        
        
       elif base_model == "ViT32-S":
             self.model = ViT_Breg(img_size=img_size, 
                    breg_dim=breg_dim, 
                    d_subs=d_subs,
                    embed_dim=384,
                    patch_size=32,
                    hidden_size=hidden_size,
                    bn=bn).to(device)
        
        
       elif base_model == "ViT32-B":
             self.model = ViT_Breg(img_size=img_size, 
                    breg_dim=breg_dim, 
                    d_subs=d_subs,
                    patch_size=32,
                    hidden_size=hidden_size,
                    bn=bn).to(device)
        
        
       elif base_model == "Swin-T":
             self.model = Swin_Breg(img_size=img_size,
                          depths=(2, 2, 6, 2),
                          breg_dim=breg_dim,
                          d_subs=d_subs,
                          hidden_size=hidden_size,
                          bn=bn).to(device)
        
        
       elif base_model == "Swin-S":
             self.model = Swin_Breg(img_size=img_size,
                          depths=(2, 2, 18, 2),
                          breg_dim=breg_dim,
                          d_subs=d_subs,
                          hidden_size=hidden_size,
                          bn=bn).to(device)
        
        
       elif base_model == "Swin-B":
             self.model = Swin_Breg(img_size=img_size,
                          depths=(2, 2, 18, 2),
                          embed_dim=128,
                          num_heads=(4, 8, 16, 32),
                          breg_dim=breg_dim,
                          d_subs=d_subs,
                          hidden_size=hidden_size,
                          bn=bn).to(device)
