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
from functools import partial
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from einops import repeat, rearrange

from terramind.models.tm_utils import (
    build_1d_sincos_posemb,
    build_2d_sincos_posemb,
    pair,
)
from terramind.models.codebook_fusion import MultiCodebookLogitsMLP, create_codebook_fusion


class SequenceDecoderEmbedding(nn.Module):
    """Embedding module for sequence inputs, like captions or a sequence of objects.

    Args:
        vocab_size: Vocabulary size
        max_length: Maximum number of tokens in the sequence
        dim_tokens: Dimension of output tokens. Can be set using init method.
        sincos_pos_emb: Set to True (default) to use fixed 1D sin-cos positional embeddings
        padding_idx: Padding index for word embedding
        share_embedding: Set to True to share input and output embedding weights
    """

    def __init__(
            self,
            vocab_size: int,
            max_length: int,
            dim_tokens: Optional[int] = None,
            sincos_pos_emb: bool = True,
            max_sincos_pos_emb: int = 512,
            padding_idx: int = 0,
            share_embedding: bool = True,
            **kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.dim_tokens = dim_tokens
        self.sincos_pos_emb = sincos_pos_emb
        self.padding_idx = padding_idx
        self.max_sincos_pos_emb = max_sincos_pos_emb
        self.share_embedding = share_embedding

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
            # Get all posembs, than truncate up to max length
            pos_emb = build_1d_sincos_posemb(
                max_len=self.max_sincos_pos_emb, embed_dim=self.dim_tokens
            )[: self.max_length]
            self.register_buffer("pos_emb", pos_emb)
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

        # Output projection layer
        self.to_logits = nn.Linear(self.dim_tokens, self.vocab_size, bias=False)

        if self.share_embedding:
            # Share input and output embedding weights
            self.to_logits.weight = self.token_emb.weight

    @torch.jit.ignore
    def no_weight_decay(self):
        return set()

    def forward_embed(self, d: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through embedding module, transforming sequence of ids to sequence of embeddings.
        Creates corresponding modality and positional embeddings and adds them to the dict.

        Args:
            d (Dict[str, torch.Tensor]): Modality dict, with at least the following keys:
                - 'tensor' (torch.Tensor): Token sequence for each batch. Shape (B, L) where B is the batch size and L is the sequence length.
                - 'target_mask' (torch.Tensor): Mask for valid tokens in the target sequence (set to 0 for valid tokens and 1 otherwise). Shape (B, L).

        Returns:
            Dict[str, torch.Tensor]: Modality dict with added keys:
                - 'x' (torch.Tensor): Embedded token sequence. Shape (B, L, D) where D is the embedding dimension.
                - 'emb' (torch.Tensor): Sum of positional and modality embeddings for the target sequence. Shape (B, L, D).
                - 'ids' (torch.Tensor): Original token sequence from input dict. Shape (B, L).
        """
        ids = d["tensor"]
        B = ids.shape[0]
        assert (
                self.dim_tokens is not None
        ), "Need to call init(dim_tokens) function first"

        # Map to embedding
        x = self.token_emb(ids)

        expanded_pos_emb = repeat(self.pos_emb, "() n d -> b n d", b=B)

        # Target pos encoding
        target_mask = d["target_mask"]
        target_pos_id = (~target_mask).int().cumsum(dim=1) - 1
        target_pos_id[target_mask] = 0
        # Sometimes target sequence is over max length, it will be truncated in decoder
        target_pos_id[target_pos_id >= self.max_length] = 0
        target_pos_emb = torch.gather(
            expanded_pos_emb,
            dim=1,
            index=repeat(target_pos_id, "b n -> b n d", d=expanded_pos_emb.shape[2]),
        )
        target_pos_emb[target_mask] = 0

        x_emb = target_pos_emb + self.mod_emb

        d["x"] = x
        d["emb"] = x_emb
        d["ids"] = d["tensor"]

        return d

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through output projection layer, transforming sequence of embeddings to logits.

        Args:
            x (torch.Tensor): Output tokens from the decoder. Shape (B, M, D)

        Returns:
            torch.Tensor: Logits for each token in the sequence. Shape (B, M, V)
        """
        logits = self.to_logits(x)
        return logits


class ImageTokenDecoderEmbedding(nn.Module):
    """Embedding module for tokenized spatial inputs.

    Args:
        vocab_size: Vocabulary size (per codebook)
        patch_size: Int or tuple of the patch size over the full image size.
        dim_tokens: Dimension of output tokens. Can be set using init method.
        sincos_pos_emb: Set to True (default) to use fixed 2D sin-cos positional embeddings
        image_size: Default image size. Used to initialize size of positional embeddings.
        share_embedding: Set to True to share input and output embedding weights
        num_codebooks: Number of codebooks (default: 1). When > 1, uses linear layer instead of embedding.
    """

    def __init__(
            self,
            vocab_size: int,
            patch_size: Union[int, Tuple[int, int]] = 16,
            dim_tokens: Optional[int] = None,
            sincos_pos_emb: bool = True,
            image_size: Union[int, Tuple[int]] = 224,
            share_embedding: bool = True,
            num_codebooks: int = 1,
            logits_hidden_ratio: int = 4,
            **kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_codebooks = num_codebooks
        self.logits_hidden_ratio = logits_hidden_ratio
        self.patch_size = pair(patch_size)
        self.dim_tokens = dim_tokens
        self.sincos_pos_emb = sincos_pos_emb
        self.image_size = pair(image_size)
        self.num_patches = (self.image_size[0] // self.patch_size[0]) * (
                self.image_size[1] // self.patch_size[1]
        )
        self.share_embedding = share_embedding

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
            self.register_buffer("pos_emb", pos_emb)
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
            # Create codebook fusion module for input embeddings
            self.codebook_fusion = create_codebook_fusion(
                fusion_type='mlp',
                dim_tokens=self.dim_tokens,
                num_codebooks=self.num_codebooks,
                hidden_ratio=4
            )
            # Use MLP-based multi-codebook logits generation
            self.to_logits = MultiCodebookLogitsMLP(
                dim_tokens=self.dim_tokens,
                num_codebooks=self.num_codebooks,
                vocab_size=self.vocab_size,
                hidden_ratio=self.logits_hidden_ratio
            )
            # Cannot share weights with MLP-based approach
            self.share_embedding = False
        else:
            # For single codebook: use standard embedding
            self.token_emb = nn.Embedding(
                num_embeddings=self.vocab_size, embedding_dim=self.dim_tokens
            )
            # Output projection layer
            self.to_logits = nn.Linear(self.dim_tokens, self.vocab_size, bias=False)

            if self.share_embedding:
                # Share input and output embedding weights
                self.to_logits.weight = self.token_emb.weight

    @torch.jit.ignore
    def no_weight_decay(self):
        return set()

    def forward_embed(self, d: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the embedding module, transforming tokenized spatial inputs to embeddings.
        Creates corresponding modality and positional embeddings and adds them to the dict.

        Args:
            d (Dict[str, torch.Tensor]): Modality dict, with at least the following key:
                - 'tensor' (torch.Tensor): Modality tokens for each batch (e.g. from tokenized images).
                  For single codebook: Shape (B, H, W) or (B, H*W)
                  For multi-codebook: Shape (B, H, W, num_codebooks) or (B, H*W, num_codebooks)

        Returns:
            Dict[str, torch.Tensor]: Modality dict with added keys:
                - 'x' (torch.Tensor): Embedded token sequence. Shape (B, H*W, D) where D is the embedding dimension.
                - 'emb' (torch.Tensor): Sum of positional and modality embeddings for the token sequence. Shape (B, H*W, D).
                - 'ids' (torch.Tensor): Reshaped token sequence from input dict. Shape (B, H*W) or (B, H*W, num_codebooks).
        """
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

            # Update ids to the flattened version for consistency
            ids = ids_flat
        else:
            # Single codebook case: ids shape is (B, H, W) or (B, num_patches)
            ids = ids.reshape(B, -1)  # (B, num_patches)
            x = self.token_emb(ids)  # (B, num_patches, dim_tokens)

        # Create positional embedding + modality embedding
        x_emb = repeat(self.pos_emb + self.mod_emb, "() n d -> b n d", b=B)

        d["x"] = x
        d["emb"] = x_emb
        d["ids"] = ids
        return d

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through output projection layer, transforming sequence of embeddings to logits.

        Args:
            x (torch.Tensor): Output tokens from the decoder. Shape (B, M, D) or (N, D)

        Returns:
            torch.Tensor: Logits for each token in the sequence.
                For single codebook: Shape (B, M, V) or (N, V) where V is vocab_size
                For multi-codebook: Shape (N, num_codebooks, V) where V is vocab_size per codebook
        """
        # The MLP-based logits generation handles both batched and flattened inputs
        # and returns the appropriate shape automatically
        logits = self.to_logits(x)

        if logits.dim() == 4:
            logits = rearrange(logits, "b l n c -> (b l) n c")

        return logits
