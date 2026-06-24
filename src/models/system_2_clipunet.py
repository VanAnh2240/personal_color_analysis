"""
src/models/system_2_clipunet.p
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import clip
except ImportError:
    raise ImportError(
        "OpenAI CLIP not installed.\n"
        "Run: pip install git+https://github.com/openai/CLIP.git"
    )

from config import LAPA_NUM_CLASSES, CLIP_MODEL_NAME, UNET_CHANNELS


class ConvBnRelu(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, pad: int = 1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, k, padding=pad, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.fuse = nn.Sequential(
            ConvBnRelu(in_ch + skip_ch, out_ch),
            ConvBnRelu(out_ch,          out_ch),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:],
                              mode="bilinear", align_corners=False)
        return self.fuse(torch.cat([x, skip], dim=1))


class ProjectionBridge(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.proj = ConvBnRelu(in_ch, out_ch, k=1, pad=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class ClipEncoder(nn.Module):
    TAPS = [3, 6, 9, 12]

    def __init__(self, model_name: str = CLIP_MODEL_NAME, freeze: bool = True):
        super().__init__()
        clip_model, _ = clip.load(model_name, device="cpu", jit=False)
        self.visual = clip_model.visual.float()

        if freeze:
            for p in self.visual.parameters():
                p.requires_grad_(False)
        self.patch_size = self.visual.conv1.kernel_size[0]   
        self.embed_dim  = self.visual.transformer.width      

    def _resize_pos_embed(self, posemb: torch.Tensor,
                          n_patches: int) -> torch.Tensor:
        D      = posemb.size(2)
        cls    = posemb[:, :1, :]        # (1, 1, D)
        grid   = posemb[:, 1:, :]        # (1, n_orig, D)
        n_orig = grid.size(1)

        if n_orig == n_patches:
            return posemb                # nothing to do

        gs_old = int(round(n_orig    ** 0.5))
        gs_new = int(round(n_patches ** 0.5))

        assert gs_old ** 2 == n_orig, (
            f"Original positional grid is not square: {n_orig} tokens "
            f"(gs_old={gs_old}, gs_old²={gs_old**2})"
        )
        assert gs_new ** 2 == n_patches, (
            f"Target patch count {n_patches} is not a perfect square. "
            f"Ensure H and W are both divisible by patch_size={self.patch_size}."
        )

        grid = (grid
                .reshape(1, gs_old, gs_old, D)
                .permute(0, 3, 1, 2))
        grid = F.interpolate(grid, size=(gs_new, gs_new),
                             mode="bilinear", align_corners=False)
        grid = (grid
                .permute(0, 2, 3, 1)
                .reshape(1, gs_new * gs_new, D))      

        return torch.cat([cls, grid], dim=1)                      
    
    def forward(self, x: torch.Tensor):
        vit = self.visual
        B, _, H, W = x.shape
        ph, pw   = H // self.patch_size, W // self.patch_size
        n_patches = ph * pw

        # ── Patch embedding ──────────────────────────────────────────────────
        tok = vit.conv1(x)                                       
        tok = tok.reshape(B, self.embed_dim, -1).permute(0, 2, 1) 

        cls = vit.class_embedding.unsqueeze(0).expand(B, -1, -1) 
        tok = torch.cat([cls, tok], dim=1)                        


        posemb = vit.positional_embedding.unsqueeze(0)            
        posemb = self._resize_pos_embed(posemb, n_patches)         
        tok    = tok + posemb.to(tok.dtype)
        tok    = vit.ln_pre(tok)

        feats = {}
        for i, blk in enumerate(vit.transformer.resblocks):
            tok = blk(tok)
            layer = i + 1
            if layer in self.TAPS:
                f = tok[:, 1:, :].permute(0, 2, 1)              
                feats[layer] = f.reshape(B, self.embed_dim, ph, pw)

        lf0 = feats[3]   # very shallow
        lf1 = feats[6]   # low-level
        lf2 = feats[9]   # mid-level
        hlf = feats[12]  # deep semantic (bottleneck)

        return hlf, lf0, lf1, lf2


class ClipUNet(nn.Module):
    def __init__(self, num_classes: int = LAPA_NUM_CLASSES,
                 freeze_clip: bool = True):
        super().__init__()
        C = UNET_CHANNELS   # [256, 128, 64, 32]

        self.encoder = ClipEncoder(freeze=freeze_clip)
        D = self.encoder.embed_dim                  
        self.proj_hlf = ProjectionBridge(D, C[0]) 
        self.proj_lf2 = ProjectionBridge(D, C[0])  
        self.proj_lf1 = ProjectionBridge(D, C[1])   
        self.proj_lf0 = ProjectionBridge(D, C[2])  

        self.up1 = UpBlock(C[0], C[0], C[1])   
        self.up2 = UpBlock(C[1], C[1], C[2])   
        self.up3 = UpBlock(C[2], C[2], C[3])   

        self.final_up = nn.Upsample(scale_factor=2, mode="bilinear",
                                    align_corners=False)
        self.seg_head = nn.Conv2d(C[3], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[-2:]

        hlf, lf0, lf1, lf2 = self.encoder(x)

        p_hlf = self.proj_hlf(hlf)
        p_lf2 = self.proj_lf2(lf2)  
        p_lf1 = self.proj_lf1(lf1)  
        p_lf0 = self.proj_lf0(lf0)  
        
        d1 = self.up1(p_hlf, p_lf2)  
        d2 = self.up2(d1,    p_lf1)  
        d3 = self.up3(d2,    p_lf0) 

        out = self.final_up(d3)       
        out = self.seg_head(out)      

        if out.shape[-2:] != (H, W):
            out = F.interpolate(out, size=(H, W),
                                mode="bilinear", align_corners=False)
        return out

if __name__ == "__main__":
    model = ClipUNet(num_classes=LAPA_NUM_CLASSES, freeze_clip=True)
    dummy = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        out = model(dummy)
    print(f"Output shape : {out.shape}")  
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    total     = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Trainable    : {trainable:.1f}M / {total:.1f}M total")