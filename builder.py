
from models import ViT_Breg, Swin_Breg

class Builder():
    def __init__(self, base_model, img_size, breg_dim, num_classes,
                 d_subs, hidden_size, bn, device):
        
       if base_model == "ViT16-S":
           self.model = ViT_Breg(img_size=img_size, 
                    breg_dim=breg_dim, 
                    d_subs=d_subs,
                    embed_dim=384,
                    num_classes=num_classes,
                    hidden_size=hidden_size,
                    bn=bn).to(device)
        
        
       elif base_model == "ViT16-B":
             self.model = ViT_Breg(img_size=img_size,
                    breg_dim=breg_dim, 
                    d_subs=d_subs,
                    num_classes=num_classes,
                    hidden_size=hidden_size,
                    bn=bn).to(device)
        
        
       elif base_model == "ViT32-S":
             self.model = ViT_Breg(img_size=img_size, 
                    breg_dim=breg_dim, 
                    d_subs=d_subs,
                    embed_dim=384,
                    num_classes=num_classes,
                    patch_size=32,
                    hidden_size=hidden_size,
                    bn=bn).to(device)
        
        
       elif base_model == "ViT32-B":
             self.model = ViT_Breg(img_size=img_size, 
                    breg_dim=breg_dim, 
                    d_subs=d_subs,
                    num_classes=num_classes,
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
                          num_classes=num_classes,
                          d_subs=d_subs,
                          hidden_size=hidden_size,
                          bn=bn).to(device)
        
        
       elif base_model == "Swin-B":
             self.model = Swin_Breg(img_size=img_size,
                          depths=(2, 2, 18, 2),
                          embed_dim=128,
                          num_heads=(4, 8, 16, 32),
                          mlp_dim=1024,
                          breg_dim=breg_dim,
                          num_classes=num_classes,
                          d_subs=d_subs,
                          hidden_size=hidden_size,
                          bn=bn).to(device)
