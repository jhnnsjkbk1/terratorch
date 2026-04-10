
import torch
import warnings
import numpy as np
from torch import nn
from einops import rearrange
import torch.nn.functional as F


class PerceptualEncoder(nn.Module):
    def __init__(self, model, pretrained, patch_size):
        super(PerceptualEncoder, self).__init__()
        try:
            import core.vision_encoder.pe as pe
            import core.vision_encoder.transforms as transforms

        except ImportError as e:
            warnings.warn(
                "Install PE with `pip install git+https://github.com/facebookresearch/perception_models.git` or "
                "`pip install ftfy && git clone https://github.com/facebookresearch/perception_models.git && "
                "mv perception_models/core core && rm -rf perception_models`")
            raise e

        self.encoder = pe.VisionTransformer.from_config(model, pretrained=pretrained).eval()
        self.model_patch_size = self.encoder.conv1.kernel_size[0]
        self.patch_size = patch_size

    def forward(self, x):
        if self.patch_size != self.model_patch_size:
            x = F.interpolate(x, scale_factor=self.model_patch_size / self.patch_size,
                              mode="bilinear", align_corners=True)
        B, C, H, W = x.shape
        N_H, N_W = H // self.model_patch_size, W // self.model_patch_size
        x = self.encoder(x)
        if N_H * N_W != x.shape[1]:
            x = x[:, 1:]  # Remove CLS token
        x = rearrange(x, "b (nh nw) d -> b d nh nw", nh=N_H, nw=N_W)

        return x


class DINOv3(nn.Module):
    def __init__(self, model, ckpt_path, pretrained, global_token=False):
        super(DINOv3, self).__init__()

        self.encoder = torch.hub.load("facebookresearch/dinov3", model, source="github",
                                      pretrained=pretrained, weights=ckpt_path).eval()
        self.model_patch_size = self.encoder.patch_embed.proj.kernel_size[0]
        self.global_token = global_token

    def forward(self, x):
        B, C, H, W = x.shape
        N_H, N_W = H // self.model_patch_size, W // self.model_patch_size

        x = self.encoder(x, is_training=True) # is_training=False returns only the cls token

        if self.global_token:
            x = x["x_norm_clstoken"][..., None, None]  # B, C, 1, 1
        else:
            x = x["x_norm_patchtokens"]
            x = rearrange(x, "b (nh nw) d -> b d nh nw", nh=N_H, nw=N_W)

        return x