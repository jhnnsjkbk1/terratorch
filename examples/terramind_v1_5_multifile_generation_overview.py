#!/usr/bin/env python3
"""
TerraMind v1.5 Multi-File Generation Overview

This script generates a comprehensive overview plot showing TerraMind v1.5 model 
generations for multiple S2L2A input files across different target modalities.

Layout: 5 rows (files) × 9 columns (S2L2A Input + 2 per modality × 4 modalities)
Target Modalities: DEM, NDVI, LULC, S1RTC
"""

import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import rioxarray as rxr

# Add paths
sys.path.insert(0, '/Users/jja/Documents/02_EOFM/TerraMind-Pretraining')
sys.path.insert(0, '/Users/jja/Documents/02_EOFM/terratorch')

from terratorch.models.backbones.terramind.model.terramind_register import (
    v1_5_pretraining_mean, 
    v1_5_pretraining_std,
    tokenizer_dict
)
from terramind.vq import get_image_tokenizer
from terratorch.models.backbones.terramind.tokenizer.tokenizer_register import (
    terramind_v1_5_tokenizer_s1rtc,
    terramind_v1_5_tokenizer_s2l2a,
    terramind_v1_5_tokenizer_dem,
    terramind_v1_5_tokenizer_ndvi,
    terramind_v1_5_tokenizer_lulc,
    terramind_v1_5_tokenizer_pe,
    terramind_v1_5_tokenizer_dinov3_lvd,
    terramind_v1_5_tokenizer_dinov3_sat,
)
from terratorch.models.backbones.terramind.model.terramind_generation import TerraMindGeneration
from plotting_utils import plot_modality, pca_visualize


def setup_device():
    """Select the best available device."""
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f"Using device: {device}")
    return device


def load_tokenizer(modality, device, dino_ckpt_path=None):
    """Load tokenizer for a specific modality."""
    tokenizer_build = {
        'S1RTC': terramind_v1_5_tokenizer_s1rtc,
        'S2L2A': terramind_v1_5_tokenizer_s2l2a,
        'NDVI': terramind_v1_5_tokenizer_ndvi,
        'DEM': terramind_v1_5_tokenizer_dem,
        'LULC': terramind_v1_5_tokenizer_lulc,
        'PE': terramind_v1_5_tokenizer_pe,
        'DINOv3_LVD': lambda **kwargs: terramind_v1_5_tokenizer_dinov3_lvd(dino_ckpt_path=dino_ckpt_path, **kwargs),
        'DINOv3_SAT': lambda **kwargs: terramind_v1_5_tokenizer_dinov3_sat(dino_ckpt_path=dino_ckpt_path, **kwargs),
    }
    
    model = tokenizer_build[modality](pretrained=True)
    model.eval()
    model.to(device)
    return model


def get_standardization_params(modalities, device):
    """Get mean and std for standardization."""
    folder_to_mod = {
        'S1RTC': 'tok_sen1rtc@224',
        'S2L2A': 'tok_sen2l2a@224',
        'NDVI': 'tok_ndvi@224',
        'DEM': 'tok_dem@224',
        'LULC': 'tok_lulc@224',
        'PE': 'tok_pe_spatial@224',
        'DINOv3_LVD': 'tok_dinov3_7b_lvd@224',
        'DINOv3_SAT': 'tok_dinov3_7b_sat@224',
    }
    
    mean = {
        modality: torch.tensor(
            v1_5_pretraining_mean[folder_to_mod[modality]],
            device=device
        )[None, :, None, None]
        for modality in modalities
    }
    
    std = {
        modality: torch.tensor(
            v1_5_pretraining_std[folder_to_mod[modality]],
            device=device
        )[None, :, None, None]
        for modality in modalities
    }
    
    return mean, std


def load_and_prepare_image(file_path, modality, mean, std, device):
    """Load and prepare an image for processing."""
    image = rxr.open_rasterio(file_path)
    data = torch.Tensor(image.values).unsqueeze(0)
    
    # Extract 224x224 patch
    data_patch = data[0, :, :224, :224].to(device)
    
    # Standardize
    data_standardized = (data_patch - mean[modality]) / std[modality]
    
    return data_patch, data_standardized


def one_hot_encode_lulc(lulc_data, num_classes=10):
    """
    One-hot encode LULC data for tokenizer.
    
    Args:
        lulc_data: Tensor of shape [C, H, W] where C=1 containing class indices
        num_classes: Number of LULC classes (default: 10)
    
    Returns:
        One-hot encoded tensor of shape [num_classes, H, W]
    """
    # Ensure we have the right shape
    if lulc_data.shape[0] == 1:
        lulc_data = lulc_data[0]  # Remove channel dim if present
    
    # Convert to long for one-hot encoding
    lulc_indices = lulc_data.long()
    
    # Create one-hot encoding
    one_hot = torch.nn.functional.one_hot(lulc_indices, num_classes=num_classes)
    
    # Transpose to [num_classes, H, W]
    one_hot = one_hot.permute(0, 3, 1, 2).float()
    
    return one_hot


def main():
    # Configuration
    device = setup_device()
    
    # File list
    file_names = [
        "637U_59R_1_3.tif",
        "38D_378R_2_3.tif",
        "282D_485L_3_3.tif",
        "433D_629L_3_1.tif",
        "609U_541L_3_0.tif",
    ]
    
    # Paths
    data_dir = Path("/Users/jja/Documents/02_EOFM/terramind/examples")
    # checkpoint_path = '/Users/jja/Documents/02_EOFM/TerraMind-Pretraining/checkpoints/v1_5/baseline/checkpoint_14.pth'
    checkpoint_path = '/Users/jja/Documents/02_EOFM/TerraMind-Pretraining/checkpoints/v1_5/pe_dino_ablation/checkpoint_10.pth'

    # Modalities
    input_modality = "S2L2A"
    target_modalities = ["DEM", "NDVI", "LULC", "S1RTC", "PE", "DINOv3_LVD", "DINOv3_SAT"]
    all_modalities = [input_modality] + target_modalities
    
    print(f"\nProcessing {len(file_names)} files with {len(target_modalities)} target modalities")
    print(f"Target modalities: {', '.join(target_modalities)}")
    
    # Get standardization parameters
    mean, std = get_standardization_params(all_modalities, device)
    
    # Load tokenizers
    print("\nLoading tokenizers...")
    tokenizers = {}
    for modality in all_modalities:
        print(f"  Loading {modality} tokenizer...")
        tokenizers[modality] = load_tokenizer(modality, device)
    
    # Load TerraMind v1.5 generation model
    print("\nLoading TerraMind v1.5 generation model...")
    model = TerraMindGeneration(
        img_size=224,
        modalities=[input_modality],
        output_modalities=target_modalities,
        tokenizer_dict=tokenizer_dict['v1_5'],
        encoder_depth=12,
        decoder_depth=4,
        dim=192,
        num_heads=3,
        decoding_steps=1,
        temps=1.0,
        top_p=0.8,
        top_k=0,
        patch_size=16,
        standardize=True,
        pretrained=True,
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    missing_keys, unexpected_keys = model.sampler.model.load_state_dict(
        checkpoint['model'], strict=False
    )
    print(f"Model loaded. Missing keys: {len(missing_keys)}")
    
    model.eval()
    model.to(device)
    
    # Create figure
    n_rows = len(file_names)
    n_cols = 1 + len(target_modalities) * 2  # Input + (Tokenizer + Generated) per modality
    
    fig, axes = plt.subplots(
        n_rows, n_cols, 
        figsize=(n_cols * 4, n_rows * 4),
        squeeze=False
    )
    
    print(f"\nCreating {n_rows}×{n_cols} subplot grid...")
    
    # Process each file
    for row_idx, file_name in enumerate(file_names):
        print(f"\nProcessing file {row_idx + 1}/{len(file_names)}: {file_name}")
        
        # Load input S2L2A
        input_path = data_dir / input_modality / file_name
        if not input_path.exists():
            print(f"  WARNING: File not found: {input_path}")
            continue
        
        input_data, input_standardized = load_and_prepare_image(
            input_path, input_modality, mean, std, device
        )
        
        # Plot input S2L2A
        col_idx = 0
        plot_modality(
            input_modality,
            input_data.cpu().numpy(),
            ax=axes[row_idx, col_idx],
            title='Input\nS2L2A' if row_idx == 0 else ''
        )
        # Add file name as y-axis label for first column
        if col_idx == 0:
            axes[row_idx, col_idx].set_ylabel(file_name, fontsize=10, rotation=0,
                                               ha='right', va='center', labelpad=10)
        col_idx += 1
        
        # Generate all target modalities
        print(f"  Generating target modalities...")
        with torch.no_grad():
            generated = model(input_standardized, verbose=False)
        
        # Process each target modality
        for target_modality in target_modalities:
            print(f"    Processing {target_modality}...")
            
            # Skip if tokenizer not loaded
            if target_modality not in tokenizers:
                axes[row_idx, col_idx].text(
                    0.5, 0.5, f'{target_modality}\nnot available',
                    ha='center', va='center',
                    fontsize=10, color='gray'
                )
                axes[row_idx, col_idx].axis('off')
                if row_idx == 0:
                    axes[row_idx, col_idx].set_title(f'Reconstructed\n{target_modality}')
                col_idx += 1
                
                axes[row_idx, col_idx].text(
                    0.5, 0.5, f'{target_modality}\nnot available',
                    ha='center', va='center',
                    fontsize=10, color='gray'
                )
                axes[row_idx, col_idx].axis('off')
                if row_idx == 0:
                    axes[row_idx, col_idx].set_title(f'Generated\n{target_modality}')
                col_idx += 1
                continue
            
            # Feature-based modalities (PE, DINOv3) don't have ground truth files
            # They are generated from the input image through feature encoders
            is_feature_modality = target_modality in ['PE', 'DINOv3_LVD', 'DINOv3_SAT']
            
            # Check if target file exists for tokenizer reconstruction
            target_path = data_dir / target_modality / file_name
            
            if target_path.exists() and not is_feature_modality:
                # Load target for tokenizer reconstruction
                target_data, target_standardized = load_and_prepare_image(
                    target_path, target_modality, mean, std, device
                )
                
                # Special handling for LULC: one-hot encode
                if target_modality == 'LULC':
                    # One-hot encode the LULC data (expects 10 classes)
                    target_standardized = one_hot_encode_lulc(
                        target_standardized, num_classes=10
                    )
                
                # Tokenize and reconstruct
                with torch.no_grad():
                    tokens = tokenizers[target_modality].tokenize(target_standardized)
                    reconstruction = tokenizers[target_modality].decode_tokens(tokens)
                    reconstruction_unstd = (
                        reconstruction * std[target_modality] + mean[target_modality]
                    )[0]
                
                # Plot tokenizer reconstruction
                plot_modality(
                    target_modality,
                    reconstruction_unstd.cpu().numpy(),
                    ax=axes[row_idx, col_idx],
                    title=f'Reconstructed\n{target_modality}' if row_idx == 0 else ''
                )
            elif is_feature_modality:
                # Feature modalities: tokenize input through the feature encoder
                # For PE and DINOv3, the tokenizer includes the feature encoder
                try:
                    with torch.no_grad():
                        # Both PE and DINOv3 need RGB input (bands 3,2,1 from S2L2A)
                        rgb_data = input_data[[3, 2, 1]]
                        rgb_std = (rgb_data - mean[target_modality]) / std[target_modality]
                        
                        # Tokenize through feature encoder and reconstruct
                        tokens = tokenizers[target_modality].tokenize(rgb_std)
                        reconstruction = tokenizers[target_modality].decode_tokens(tokens)
                        
                        # Visualize the reconstruction using PCA (features -> RGB)
                        recon_vis = pca_visualize(reconstruction[0], n_components=3)
                        axes[row_idx, col_idx].imshow(recon_vis)
                        axes[row_idx, col_idx].axis('off')
                        if row_idx == 0:
                            axes[row_idx, col_idx].set_title(f'Reconstructed\n{target_modality}')
                except Exception as e:
                    print(f"      Warning: Could not reconstruct {target_modality}: {e}")
                    axes[row_idx, col_idx].text(
                        0.5, 0.5, 'Reconstruction\nfailed',
                        ha='center', va='center',
                        fontsize=10, color='orange'
                    )
                    axes[row_idx, col_idx].axis('off')
                    if row_idx == 0:
                        axes[row_idx, col_idx].set_title(f'Reconstructed\n{target_modality}')
            else:
                # No target file available - leave blank or show message
                axes[row_idx, col_idx].text(
                    0.5, 0.5, 'No target\navailable',
                    ha='center', va='center',
                    fontsize=10, color='gray'
                )
                axes[row_idx, col_idx].axis('off')
                if row_idx == 0:
                    axes[row_idx, col_idx].set_title(f'Reconstructed\n{target_modality}')
            
            col_idx += 1
            
            # Plot generated output
            if target_modality in generated:
                # For feature modalities, don't unstandardize (they're in feature space)
                # For other modalities, unstandardize to original scale
                if is_feature_modality:
                    # Features are already in the right space, just visualize with PCA
                    gen_vis = pca_visualize(generated[target_modality][0], n_components=3)
                    axes[row_idx, col_idx].imshow(gen_vis)
                    axes[row_idx, col_idx].axis('off')
                    if row_idx == 0:
                        axes[row_idx, col_idx].set_title(f'Generated\n{target_modality}')
                else:
                    # Unstandardize for regular modalities
                    generated_unstd = (
                        generated[target_modality][0] * std[target_modality] +
                        mean[target_modality]
                    )
                    generated_np = generated_unstd.detach().cpu().numpy()
                    plot_modality(
                        target_modality,
                        generated_np,
                        ax=axes[row_idx, col_idx],
                        title=f'Generated\n{target_modality}' if row_idx == 0 else ''
                    )
            else:
                axes[row_idx, col_idx].text(
                    0.5, 0.5, 'Generation\nfailed',
                    ha='center', va='center',
                    fontsize=10, color='red'
                )
                axes[row_idx, col_idx].axis('off')
                if row_idx == 0:
                    axes[row_idx, col_idx].set_title(f'Generated\n{target_modality}')
            
            col_idx += 1
    
    # Adjust layout and save
    plt.tight_layout()
    output_path = 'terramind_v1_5_multifile_generation_overview.pdf'
    print(f"\nSaving figure to {output_path}...")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Done! Figure saved successfully.")
    
    # Also show the plot
    plt.show()


if __name__ == "__main__":
    main()

# Made with Bob
