# Copyright 2024 EPFL and Apple Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
from einops import rearrange, repeat

from terramind.models.tm_utils import (
    build_1d_sincos_posemb,
    build_2d_sincos_posemb,
    pair,
)
from terramind.models.codebook_fusion import create_codebook_fusion


class SequenceEncoderEmbedding(nn.Module):
    """Embedding module for encoding sequence inputs, like captions or a sequence of objects.

    Args:
        vocab_size: Vocabulary size
        max_length: Maximum number of tokens in the sequence
        dim_tokens: Dimension of output tokens. Can be set using init method.
        sincos_pos_emb: Set to True (default) to use fixed 1D sin-cos positional embeddings
        max_sincos_pos_emb: Maximum allowed length for sin-cos positional embeddings
        padding_idx: Padding index for word embedding
    """

    def __init__(
            self,
            vocab_size: int,
            max_length: int,
            dim_tokens: int | None = None,
            sincos_pos_emb: bool = True,
            max_sincos_pos_emb: int = 512,
            padding_idx: int = 0,
            **kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.dim_tokens = dim_tokens
        self.sincos_pos_emb = sincos_pos_emb
        self.padding_idx = padding_idx
        self.max_sincos_pos_emb = max_sincos_pos_emb

        if self.dim_tokens is not None:
            self.init(dim_tokens=dim_tokens)

    def init(self, dim_tokens: int = 768, init_std=0.02):
        """
        Initialize parts of embedding module that are dependent on dimension of tokens.
        Should be called when setting up FourM.

        Args:
            dim_tokens: Dimension of tokens
            init_std: Standard deviation of init
        """

        self.dim_tokens = dim_tokens

        # Task embedding identifying from which task a given token comes from
        # Fixed-size positional embeddings. Can be interpolated to different input sizes
        if self.sincos_pos_emb:
            if self.max_length > self.max_sincos_pos_emb:
                raise ValueError(
                    f"Max length ({self.max_length}) is greater than the number of posembs ({self.max_sincos_pos_emb}"
                )
            pos_emb = build_1d_sincos_posemb(
                max_len=self.max_sincos_pos_emb, embed_dim=self.dim_tokens
            )[: self.max_length]
            self.register_buffer(
                "pos_emb", pos_emb
            )  # self.pos_emb is now a buffer for FSDP

        else:
            self.pos_emb = nn.Parameter(
                torch.zeros(1, self.max_length, self.dim_tokens)
            )
            nn.init.normal_(self.pos_emb, std=init_std)

        self.mod_emb = nn.Parameter(torch.zeros(1, 1, self.dim_tokens))
        nn.init.normal_(self.mod_emb, std=init_std)

        # Token embedding
        self.token_emb = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.dim_tokens,
            padding_idx=self.padding_idx,
        )

    @torch.jit.ignore
    def no_weight_decay(self):
        return set()

    def forward(
            self, d: torch.Tensor | dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass through embedding module, transforming sequence of ids to sequence of embeddings.
        Creates corresponding modality and positional embeddings and adds them to the dict.

        Args:
            d (dict[str, torch.Tensor]): Modality dict with at least the following keys:
                - 'tensor' (torch.Tensor): Input token sequence for each batch. Shape (B, L) where B is the batch size and L is the sequence length.
                - 'input_mask' (torch.Tensor): Mask for valid tokens in the input sequence (set to 0 for valid tokens and 1 otherwise). Shape (B, L).

        Returns:
            dict[str, torch.Tensor]: Modality dict with added keys:
                - 'x' (torch.Tensor): Embedded token sequence. Shape (B, L, D) where D is the embedding dimension.
                - 'emb' (torch.Tensor): Sum of positional and modality embeddings for the input sequence. Shape (B, L, D).
        """
        if not isinstance(d, dict):
            d = {
                "tensor": d,
                "input_mask": torch.zeros_like(d, dtype=torch.bool),  # No masking
            }

        ids = d["tensor"]
        B = ids.shape[0]
        assert (
                self.dim_tokens is not None
        ), "Need to call init(dim_tokens) function first"

        # Map to embedding
        x = self.token_emb(ids)

        expanded_pos_emb = repeat(self.pos_emb, "() n d -> b n d", b=B)
        # Input pos encoding
        input_mask = d["input_mask"]
        input_pos_id = (~input_mask).int().cumsum(dim=1) - 1
        input_pos_id[input_mask] = 0
        input_pos_emb = torch.gather(
            expanded_pos_emb,
            dim=1,
            index=repeat(input_pos_id, "b n -> b n d", d=expanded_pos_emb.shape[2]),
        )
        input_pos_emb[input_mask] = 0

        x_emb = input_pos_emb + self.mod_emb

        d["x"] = x
        d["emb"] = x_emb
        return d


class ImageTokenEncoderEmbedding(nn.Module):
    """Embedding module for tokenized spatial inputs.

    Args:
        vocab_size: Vocabulary size (per codebook)
        patch_size: Int or tuple of the patch size over the full image size.
        dim_tokens: Dimension of output tokens. Can be set using init method.
        sincos_pos_emb: Set to True (default) to use fixed 2D sin-cos positional embeddings
        image_size: Default image size. Used to initialize size of positional embeddings.
        num_codebooks: Number of codebooks (default: 1). When > 1, uses codebook fusion.
        fusion_type: Type of fusion for multi-codebook ('attention', 'weighted', or 'mlp'). Default: 'attention'
        fusion_num_heads: Number of attention heads for attention fusion. Default: 8
        fusion_hidden_ratio: Hidden dimension ratio for MLP fusion. Default: 4
    """

    def __init__(
            self,
            vocab_size: int,
            patch_size: int | tuple[int, int] = 16,
            dim_tokens: int | None = None,
            sincos_pos_emb: bool = True,
            image_size: int | tuple[int] = 224,
            num_codebooks: int = 1,
            fusion_type: str = 'mlp',
            fusion_num_heads: int = 8,
            fusion_hidden_ratio: int = 4,
            **kwargs,
    ):

        super().__init__()
        self.vocab_size = vocab_size
        self.num_codebooks = num_codebooks
        self.fusion_type = fusion_type
        self.fusion_num_heads = fusion_num_heads
        self.fusion_hidden_ratio = fusion_hidden_ratio
        self.patch_size = pair(patch_size)
        self.dim_tokens = dim_tokens
        self.sincos_pos_emb = sincos_pos_emb
        self.image_size = pair(image_size)
        self.num_patches = (self.image_size[0] // patch_size) * (
                self.image_size[1] // patch_size
        )

        if self.dim_tokens is not None:
            self.init(dim_tokens=dim_tokens)

    def init(self, dim_tokens: int = 768, init_std=0.02):
        """
        Initialize parts of module that are dependent on dimension of tokens.
        Should be called when setting up FourM.

        Args:
            dim_tokens: Dimension of tokens
            init_std: Standard deviation of init
        """
        self.dim_tokens = dim_tokens

        # Task embedding identifying from which task a given token comes from
        # Fixed-size positional embeddings. Can be interpolated to different input sizes
        h_posemb = self.image_size[0] // self.patch_size[0]
        w_posemb = self.image_size[1] // self.patch_size[1]
        if self.sincos_pos_emb:
            pos_emb = build_2d_sincos_posemb(
                h=h_posemb, w=w_posemb, embed_dim=self.dim_tokens
            )
            self.register_buffer(
                "pos_emb", pos_emb
            )  # self.pos_emb is now a buffer for FSDP
        else:
            self.pos_emb = nn.Parameter(
                torch.zeros(1, (h_posemb * w_posemb), self.dim_tokens)
            )
            nn.init.normal_(self.pos_emb, std=init_std)

        self.mod_emb = nn.Parameter(torch.zeros(1, 1, self.dim_tokens))
        nn.init.normal_(self.mod_emb, std=init_std)

        # Token embedding: use codebook fusion for multi-codebook, embedding for single codebook
        if self.num_codebooks > 1:
            # For multi-codebook: embed each codebook to dim_tokens, then fuse
            self.token_emb = nn.Embedding(
                num_embeddings=self.vocab_size, embedding_dim=self.dim_tokens
            )
            # Create codebook fusion module
            self.codebook_fusion = create_codebook_fusion(
                fusion_type=self.fusion_type,
                dim_tokens=self.dim_tokens,
                num_codebooks=self.num_codebooks,
                num_heads=self.fusion_num_heads,
                hidden_ratio=self.fusion_hidden_ratio
            )
        else:
            # For single codebook: use standard embedding
            self.token_emb = nn.Embedding(
                num_embeddings=self.vocab_size, embedding_dim=self.dim_tokens
            )

    @torch.jit.ignore
    def no_weight_decay(self):
        return set()

    def forward(
            self, d: torch.Tensor | dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass through embedding module, transforming image tokens to a sequence of embeddings.
        Creates corresponding modality and positional embeddings and adds them to the dict.

        Args:
            d (torch.Tensor, dict[str, torch.Tensor]): Modality dict with at least the following key:
                - 'tensor' (torch.Tensor): Input image tokens for each batch.
                  For single codebook: Shape (B, H, W) or (B, H*W)
                  For multi-codebook: Shape (B, H, W, num_codebooks) or (B, H*W, num_codebooks)

        Returns:
            dict[str, torch.Tensor]: Modality dictionary with added keys:
                - 'x' (torch.Tensor): Embedded token sequence. Shape (B, H*W, D).
                - 'emb' (torch.Tensor): Sum of positional and modality embeddings for the input sequence. Shape (B, H*W, D).
        """
        if not isinstance(d, dict):
            d = {"tensor": d}

        ids = d["tensor"]
        B = ids.shape[0]

        if self.num_codebooks > 1:
            # Multi-codebook case: ids shape is (B, H, W, num_codebooks) or (B, num_patches, num_codebooks)
            # Using codebook fusion approach:
            # 1. Embed each codebook token to dim_tokens independently
            # 2. Apply codebook fusion to aggregate embeddings

            # Reshape to (B, num_patches, num_codebooks) if needed
            if len(ids.shape) == 4:
                # (B, H, W, num_codebooks)
                ids_flat = ids.reshape(B, -1, self.num_codebooks)  # (B, H*W, num_codebooks)
            else:
                # (B, num_patches, num_codebooks)
                ids_flat = ids

            num_patches = ids_flat.shape[1]

            # Step 1: Embed each codebook token independently to dim_tokens
            # ids_flat: (B, num_patches, num_codebooks) with values 0 to vocab_size-1
            # Reshape to (B * num_patches * num_codebooks) for embedding lookup
            ids_for_emb = ids_flat.reshape(-1)  # (B * num_patches * num_codebooks)
            emb = self.token_emb(ids_for_emb)  # (B * num_patches * num_codebooks, dim_tokens)

            # Reshape to (B, num_patches, num_codebooks, dim_tokens)
            emb = emb.reshape(B, num_patches, self.num_codebooks, self.dim_tokens)

            # Step 2: Apply codebook fusion to aggregate embeddings
            # Input: (B, num_patches, num_codebooks, dim_tokens)
            # Output: (B, num_patches, dim_tokens)
            x = self.codebook_fusion(emb)
        else:
            # Single codebook case: ids shape is (B, H, W) or (B, num_patches)
            ids = ids.reshape(B, -1)  # (B, num_patches)
            x = self.token_emb(ids)  # (B, num_patches, dim_tokens)

        # Create positional embedding + modality embedding
        x_emb = repeat(self.pos_emb + self.mod_emb, "() n d -> b n d", b=B)

        d["x"] = x
        d["emb"] = x_emb

        return d


class ImageEncoderEmbedding(nn.Module):
    """Embedding module for spatial inputs, like images or feature maps.
    Creates tokens from patches over the image.

    This adapter / embedding differs from the one of MultiMAE by taking as input a dict and

    separating positional embeddings and modality embeddings from the input projection

    Input projection is 'x', posemb + modemb is 'emb'

    Args:
        num_channels: Number of input channels of the image/feature map
        patch_size: Int or tuple of the patch size over the full image size.
        dim_tokens: Dimension of output tokens. Can be set using init method.
        sincos_pos_emb: Set to True (default) to use fixed 2D sin-cos positional embeddings
        image_size: Default image size. Used to initialize size of positional embeddings.
    """

    def __init__(
            self,
            num_channels: int,
            patch_size: int | tuple[int, int],
            dim_tokens: int | None = None,
            sincos_pos_emb: bool = True,
            image_size: int | tuple[int] = 224,
            **kwargs,
    ):

        super().__init__()
        self.num_channels = num_channels
        self.patch_size = pair(patch_size)
        self.dim_tokens = dim_tokens
        self.sincos_pos_emb = sincos_pos_emb
        self.image_size = pair(image_size)
        self.num_patches = (self.image_size[0] // patch_size) * (
                self.image_size[1] // patch_size
        )

        if self.dim_tokens is not None:
            self.init(dim_tokens=dim_tokens)

    def init(self, dim_tokens: int = 768, init_std=0.02):
        """
        Initialize parts of encoder that are dependent on dimension of tokens.
        Should be called when setting up FourM.

        Args:
            dim_tokens: Dimension of tokens
            init_std: Standard deviation of init
        """
        self.dim_tokens = dim_tokens

        # Task embedding identifying from which task a given token comes from
        # Fixed-size positional embeddings. Can be interpolated to different input sizes
        h_posemb = self.image_size[0] // self.patch_size[0]
        w_posemb = self.image_size[1] // self.patch_size[1]
        if self.sincos_pos_emb:
            pos_emb = build_2d_sincos_posemb(
                h=h_posemb, w=w_posemb, embed_dim=self.dim_tokens
            )
            self.register_buffer(
                "pos_emb", pos_emb
            )  # self.pos_emb is now a buffer for FSDP
        else:
            self.pos_emb = nn.Parameter(
                torch.zeros(1, (h_posemb * w_posemb), self.dim_tokens)
            )
            nn.init.normal_(self.pos_emb, std=init_std)

        self.mod_emb = nn.Parameter(torch.zeros(1, 1, self.dim_tokens))
        nn.init.normal_(self.mod_emb, std=init_std)

        # Image -> tokens projection
        # No bias term here, so modality embedding fully comes from self.mod_emb
        self.proj = nn.Linear(
            self.num_channels * self.patch_size[0] * self.patch_size[1],
            self.dim_tokens,
            bias=False,
        )

    @torch.jit.ignore
    def no_weight_decay(self):
        return set()

    def forward(
            self, d: torch.Tensor | dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass through embedding module, transforming image to sequence of tokens.
        Creates corresponding modality and positional embeddings and adds them to the dict.

        Args:
            d (torch.Tensor, dict[str, torch.Tensor]): Modality dict with at least the following key:
                - 'tensor' (torch.Tensor): Input image for each batch. Shape (B, C, H, W) where B is the batch size, C is the number of channels, and H, W are height and width of the image.

        Returns:
            dict[str, torch.Tensor]: Modality dict with added keys:
                - 'x' (torch.Tensor): Embedded token sequence. Shape (B, (H / PH) * (W / PW), D), where PH and PW are the patch sizes
                - 'emb' (torch.Tensor): Sum of positional and modality embeddings for the input sequence. Shape (B, (H / PH) * (W / PW), D)
        """
        if not isinstance(d, dict):
            d = {"tensor": d}

        x = d["tensor"]
        B, C, H, W = x.shape
        assert (
                self.dim_tokens is not None
        ), "Need to call init(dim_tokens) function first"
        assert (H % self.patch_size[0] == 0) and (
                W % self.patch_size[1] == 0
        ), f"Image sizes {H}x{W} must be divisible by patch sizes {self.patch_size[0]}x{self.patch_size[1]}"

        # Create patches [B, C, H, W] -> [B, N_patches, ph*pw*C]
        # Then project to [B, N_patches, dim_tokens]
        x_patch = self.proj(
            rearrange(
                x,
                "b d (nh ph) (nw pw) -> b (nh nw) (ph pw d)",
                ph=self.patch_size[0],
                pw=self.patch_size[1],
            )
        )

        if (H, W) != self.image_size:
            # Interpolate embedding if required
            pos_emb = self.interpolate_pos_encoding(self.pos_emb.clone(), H, W)
        else:
            pos_emb = self.pos_emb

        # Create positional embedding + modality embedding
        x_emb = repeat(pos_emb + self.mod_emb, "() n d -> b n d", b=B)

        d["x"] = x_patch
        d["emb"] = x_emb

        return d

    def interpolate_pos_encoding(
            self, pos_embeddings: torch.Tensor, height, width
    ) -> torch.Tensor:
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher resolution
        images. This method is also adapted to support torch.jit tracing.

        Adapted from:
        - transformers.models.vit.modeling_vit.ViTEmbeddings.interpolate_pos_encoding
        - https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174-L194, and
        - https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/models/vision_transformer.py#L179-L211
        """

        num_positions = pos_embeddings.shape[1]
        new_height = height // self.patch_size[0]
        new_width = width // self.patch_size[1]

        # Assuming squared default image size
        sqrt_num_positions = int(num_positions ** 0.5)
        pos_embeddings = pos_embeddings.reshape(
            1, sqrt_num_positions, sqrt_num_positions, self.dim_tokens
        )
        pos_embeddings = pos_embeddings.permute(0, 3, 1, 2)

        pos_embeddings = nn.functional.interpolate(
            pos_embeddings,
            size=(new_height, new_width),
            mode="bicubic",
            align_corners=False,
        )

        pos_embeddings = pos_embeddings.permute(0, 2, 3, 1).view(1, -1, self.dim_tokens)

        return pos_embeddings
