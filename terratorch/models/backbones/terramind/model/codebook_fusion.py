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

"""
Codebook fusion modules for aggregating multiple VQ-VAE codebook embeddings
per patch into a single token representation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CodebookFusionAttention(nn.Module):
    """
    Aggregate multiple codebook embeddings per patch using attention-based pooling.
    
    This is the recommended approach as it provides maximum flexibility and allows
    the model to learn which codebooks are most important for each patch.
    
    Args:
        dim_tokens: Dimension of token embeddings
        num_codebooks: Number of codebooks to aggregate
        num_heads: Number of attention heads (default: 8)
    """
    
    def __init__(self, dim_tokens: int, num_codebooks: int, num_heads: int = 8):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.dim_tokens = dim_tokens
        
        # Learnable query for aggregation
        self.query = nn.Parameter(torch.randn(1, 1, dim_tokens))
        nn.init.normal_(self.query, std=0.02)
        
        # Multi-head attention for pooling
        self.attention = nn.MultiheadAttention(
            embed_dim=dim_tokens,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.0
        )
        
        # Layer norm for stability
        self.norm = nn.LayerNorm(dim_tokens)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Aggregate codebook embeddings using attention pooling.
        
        Args:
            x: Codebook embeddings of shape (B, num_patches, num_codebooks, D)
            
        Returns:
            Aggregated embeddings of shape (B, num_patches, D)
        """
        B, N, C, D = x.shape
        assert C == self.num_codebooks, f"Expected {self.num_codebooks} codebooks, got {C}"
        assert D == self.dim_tokens, f"Expected dim {self.dim_tokens}, got {D}"
        
        # Reshape to process all patches together
        x_flat = x.reshape(B * N, C, D)
        
        # Expand query for all patches
        query = self.query.expand(B * N, 1, D)
        
        # Attention pooling over codebooks
        # query: (B*N, 1, D), key/value: (B*N, C, D)
        aggregated, _ = self.attention(query, x_flat, x_flat)
        
        # Reshape back and normalize
        aggregated = aggregated.reshape(B, N, D)
        aggregated = self.norm(aggregated)
        
        return aggregated


class CodebookFusionWeighted(nn.Module):
    """
    Aggregate codebooks via learned weighted sum.
    
    This is a lightweight approach that learns a fixed weight for each codebook
    across all patches. More efficient than attention but less flexible.
    
    Args:
        dim_tokens: Dimension of token embeddings
        num_codebooks: Number of codebooks to aggregate
    """
    
    def __init__(self, dim_tokens: int, num_codebooks: int):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.dim_tokens = dim_tokens
        
        # Learnable weights for each codebook
        self.weights = nn.Parameter(torch.ones(num_codebooks) / num_codebooks)
        
        # Layer norm for stability
        self.norm = nn.LayerNorm(dim_tokens)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Aggregate codebook embeddings using learned weights.
        
        Args:
            x: Codebook embeddings of shape (B, num_patches, num_codebooks, D)
            
        Returns:
            Aggregated embeddings of shape (B, num_patches, D)
        """
        B, N, C, D = x.shape
        assert C == self.num_codebooks, f"Expected {self.num_codebooks} codebooks, got {C}"
        assert D == self.dim_tokens, f"Expected dim {self.dim_tokens}, got {D}"
        
        # Softmax weights to ensure they sum to 1
        weights = F.softmax(self.weights, dim=0)
        
        # Weighted sum: (B, N, C, D) * (C,) -> (B, N, D)
        aggregated = torch.einsum('bncd,c->bnd', x, weights)
        
        # Normalize
        aggregated = self.norm(aggregated)
        
        return aggregated


class CodebookFusionMLP(nn.Module):
    """
    Aggregate codebooks via MLP projection.
    
    This approach concatenates all codebook embeddings and projects them
    through an MLP. Provides good expressiveness but higher memory usage.
    
    Args:
        dim_tokens: Dimension of token embeddings
        num_codebooks: Number of codebooks to aggregate
        hidden_ratio: Ratio of hidden dimension to input dimension (default: 4)
    """
    
    def __init__(self, dim_tokens: int, num_codebooks: int, hidden_ratio: int = 4):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.dim_tokens = dim_tokens
        
        input_dim = dim_tokens * num_codebooks
        hidden_dim = dim_tokens * hidden_ratio
        
        # MLP for fusion
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, dim_tokens),
            nn.LayerNorm(dim_tokens)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Aggregate codebook embeddings using MLP projection.
        
        Args:
            x: Codebook embeddings of shape (B, num_patches, num_codebooks, D)
            
        Returns:
            Aggregated embeddings of shape (B, num_patches, D)
        """
        B, N, C, D = x.shape
        assert C == self.num_codebooks, f"Expected {self.num_codebooks} codebooks, got {C}"
        assert D == self.dim_tokens, f"Expected dim {self.dim_tokens}, got {D}"
        
        # Flatten codebooks: (B, N, C, D) -> (B, N, C*D)
        x_flat = x.reshape(B, N, C * D)
        
        # Project through MLP
        aggregated = self.mlp(x_flat)
        
        return aggregated


def create_codebook_fusion(
    fusion_type: str,
    dim_tokens: int,
    num_codebooks: int,
    **kwargs
) -> nn.Module:
    """
    Factory function to create codebook fusion module.
    
    Args:
        fusion_type: Type of fusion ('attention', 'weighted', or 'mlp')
        dim_tokens: Dimension of token embeddings
        num_codebooks: Number of codebooks to aggregate
        **kwargs: Additional arguments for specific fusion types
        
    Returns:
        Codebook fusion module
        
    Raises:
        ValueError: If fusion_type is not recognized
    """
    if fusion_type == 'attention':
        num_heads = kwargs.get('num_heads', 8)
        return CodebookFusionAttention(dim_tokens, num_codebooks, num_heads)
    elif fusion_type == 'weighted':
        return CodebookFusionWeighted(dim_tokens, num_codebooks)
    elif fusion_type == 'mlp':
        hidden_ratio = kwargs.get('hidden_ratio', 4)
        return CodebookFusionMLP(dim_tokens, num_codebooks, hidden_ratio)
    else:
        raise ValueError(
            f"Unknown fusion_type: {fusion_type}. "
            f"Must be one of: 'attention', 'weighted', 'mlp'"
        )


class MultiCodebookLogitsMLP(nn.Module):
    """
    Generate logits for multiple codebooks from a single token embedding via MLP projection.
    
    This is the decoder counterpart to CodebookFusionMLP. While fusion aggregates multiple
    codebook embeddings into one token, this module expands one token into multiple codebook logits.
    
    Architecture:
    - Input: Single token embedding (B, N, D)
    - MLP expansion: D -> hidden_dim -> (num_codebooks * vocab_size)
    - Output: Logits for each codebook (B, N, num_codebooks, vocab_size)
    
    Args:
        dim_tokens: Dimension of input token embeddings
        num_codebooks: Number of codebooks to generate logits for
        vocab_size: Vocabulary size per codebook
        hidden_ratio: Ratio of hidden dimension to output dimension (default: 4)
    """
    
    def __init__(
        self,
        dim_tokens: int,
        num_codebooks: int,
        vocab_size: int,
        hidden_ratio: int = 4
    ):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.dim_tokens = dim_tokens
        self.vocab_size = vocab_size
        
        output_dim = num_codebooks * vocab_size
        hidden_dim = output_dim * hidden_ratio
        
        # MLP for generating multiple codebook logits from single token
        self.mlp = nn.Sequential(
            nn.Linear(dim_tokens, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate multiple codebook logits from token embeddings using MLP projection.
        
        Args:
            x: Token embeddings of shape (B, num_patches, D) or (N, D) where N = B * num_patches
            
        Returns:
            Logits of shape (B, num_patches, num_codebooks, vocab_size) or (N, num_codebooks, vocab_size)
        """
        # Store original shape
        is_batched = len(x.shape) == 3
        
        if is_batched:
            B, N, D = x.shape
            assert D == self.dim_tokens, f"Expected dim {self.dim_tokens}, got {D}"
            # Flatten to (B*N, D) for processing
            x_flat = x.reshape(B * N, D)
        else:
            # Already flattened (N, D)
            N_flat, D = x.shape
            assert D == self.dim_tokens, f"Expected dim {self.dim_tokens}, got {D}"
            x_flat = x
            B = 0  # Not used in non-batched case
            N = 0  # Not used in non-batched case
        
        # Project through MLP: (B*N, D) -> (B*N, num_codebooks * vocab_size)
        logits_flat = self.mlp(x_flat)
        
        # Reshape to separate codebooks: (B*N, num_codebooks * vocab_size) -> (B*N, num_codebooks, vocab_size)
        logits = logits_flat.reshape(-1, self.num_codebooks, self.vocab_size)
        
        if is_batched:
            # Reshape back to batch format: (B*N, num_codebooks, vocab_size) -> (B, N, num_codebooks, vocab_size)
            logits = logits.reshape(B, N, self.num_codebooks, self.vocab_size)
        
        return logits

