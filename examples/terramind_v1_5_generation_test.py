#%% md
# # TerraMind v1.5 Multicodebook Generation Test
# 
# This notebook demonstrates:
# 1. Loading data from TerraMesh v1.5 zarr files
# 2. Tokenizer reconstruction with multicodebook (128 codebooks)
# 3. Full multimodal generation using TerraMind v1.5 checkpoint
# 
# ## Setup
#%%
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import rioxarray as rxr


# Add paths
sys.path.insert(0, '/Users/jja/Documents/02_EOFM/TerraMind-Pretraining')
sys.path.insert(0, '/Users/jja/Documents/02_EOFM/terratorch')

# Device selection
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

print(f"Using device: {device}")
modalities = ['S2L2A'] # 'S1RTC', 'S1RTC_tokens', "LULC_tokens", , "NDVI", "NDVI_tokens"

#%% md
# ## Load Data from TerraMesh v1.5
#%%
from terramind.data.terramesh import build_terramesh_dataset
from terramind.utils.plotting_utils import plot_modality

# Params
data_path = Path('/Users/jja/Documents/02_EOFM/TerraMind-Pretraining/data/TerraMesh_1_5_small/train')
modalities = ['S2L2A']  # 'S1RTC', 'S1RTC_tokens', 'LULC_tokens', 'NDVI', 'NDVI_tokens'
files = 'majortom_shard_000001.tar'

# Build dataset
multi_temporal = 'SSL4EOS12' in str(data_path)
if len(modalities) == 1:
    urls = str(data_path / modalities[0] / files)
else:
    urls = str(data_path / f"[{','.join(modalities)}]" / files)

dataset = build_terramesh_dataset(
    urls=urls,
    modalities=modalities,
    time_dim=multi_temporal,
    shuffle=False,
    batch_size=1
)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=None)
print(f"Dataset loaded from: {urls}")
#%% md
# ## Load TerraMind v1.5 Tokenizer (Multicodebook)
#%%
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

import zarr
from terratorch.models.backbones.terramind.model.terramind_register import v1_pretraining_mean, v1_pretraining_std
from terramind.data.terramesh import build_terramesh_dataset
from terramind.utils.plotting_utils import plot_modality
from terramind.vq import get_image_tokenizer
from terramind.vq.tokenizer_register import (
    terramind_v1_5_tokenizer_lulc,
    terramind_v1_5_tokenizer_s1rtc,
    terramind_v1_5_tokenizer_s2l2a,
    terramind_v1_5_tokenizer_dem,
    terramind_v1_5_tokenizer_ndvi,
    terramind_v1_5_tokenizer_s1grd,
    terramind_v1_5_tokenizer_s2l1c,
    terramind_v1_5_tokenizer_s2rgb
)

# v1.5 tokenizers
tokenizer_build = {
    # 'S2RGB': terramind_v1_5_tokenizer_s2rgb,
    # 'S2L1C': terramind_v1_5_tokenizer_s2l1c,
    'S2L2A': terramind_v1_5_tokenizer_s2l2a,
    # 'S1GRD': terramind_v1_5_tokenizer_s1grd,
    # 'S1RTC': terramind_v1_5_tokenizer_s1rtc,
    # 'DEM': terramind_v1_5_tokenizer_dem,
    # 'NDVI': terramind_v1_5_tokenizer_ndvi,
    # 'LULC': terramind_v1_5_tokenizer_lulc,
}

max_tokens = 8**5

# Build tokenizer
def load_model(modality, ckpt_path=None):
    if ckpt_path is not None:
        ckpt_path = Path(ckpt_path)
        model, _ = get_image_tokenizer(
            str(ckpt_path.stem),
            tokenizers_root=str(ckpt_path.parent),
            encoder_only=False,
            device=device,
        )
        model.eval()
        model.to(device)
        return model

    model = tokenizer_build[modality](pretrained=True)
    model.eval()
    model.to(device)
    return model

tokenizer = {
    modality: load_model(modality.split('_')[0])
    for modality in modalities
}

# Get first tokenizer for display
tokenizer_s2l2a = tokenizer['S2L2A']

print(f"Tokenizer loaded:")
print(f"  - Codebooks: {tokenizer_s2l2a.num_codebooks}")
print(f"  - Codebook size: {tokenizer_s2l2a.codebook_size}")
print(f"  - Latent dim: {tokenizer_s2l2a.latent_dim}")
print(f"  - Quantizer type: {tokenizer_s2l2a.quant_type}")

# Get standardization values
folder_to_mod = {
    # 'S2RGB': 'untok_sen2rgb@224',
    # 'S1GRD': 'tok_sen1grd@224',
    # 'S1RTC': 'tok_sen1rtc@224',
    'S2L2A': 'tok_sen2l2a@224',
    # 'NDVI': 'tok_ndvi@224',
    # 'DEM': 'tok_dem@224',
    # 'LULC': 'tok_lulc@224',
}
mean = {
    modality: torch.tensor(v1_pretraining_mean[folder_to_mod[modality.split('_')[0]]], device=device)[None, :, None, None]
    for modality in modalities
}
std = {
    modality: torch.tensor(v1_pretraining_std[folder_to_mod[modality.split('_')[0]]], device=device)[None, :, None, None]
    for modality in modalities
}
#%% md
# ## Test Tokenizer Reconstruction
#%%
# Get first sample
data = {}
sample = next(iter(dataloader))

for modality in modalities:
    examples = [
        # '../examples/S2L2A/38D_378R_2_3.tif',
        # '../examples/S2L2A/282D_485L_3_3.tif',
        # '../examples/S2L2A/433D_629L_3_1.tif',
        # '../examples/S2L2A/637U_59R_1_3.tif',
        # '../examples/S2L2A/609U_541L_3_0.tif',
        # '../examples/S2L2A/Frascati_S2L2A.tif',
        '/Users/jja/Documents/02_EOFM/terramind/examples/S2L2A/Langnau am Albis_S2L2A.tif',
        # '../examples/S2L2A/Langnau am Albis_S2L2A.tif',
    ]

    example_id = 0  # Select id between 0 and 4
    # Load an S-2 L2A example
    image = rxr.open_rasterio(examples[example_id])
    # Convert to shape [B, C, 224, 224]
    image = torch.Tensor(image.values, device='cpu').unsqueeze(0)


    sample['S2L2A'] = image # sample['image']

    # Extract and prepare data
    data[modality] = sample[modality][0, :, :224, :224]
    input = sample[modality][:1, :, :224, :224].to(device)

    # Standardize
    input_standardized = (input - mean[modality]) / std[modality]

    # Tokenize (multicodebook output: B, num_codebooks, H, W)
    with torch.no_grad():
        tokens = tokenizer_s2l2a.tokenize(input_standardized)
        print(f"Token shape (multicodebook): {tokens.shape}")  # Should be (1, 128, 14, 14)

        # Decode tokens
        reconstruction = tokenizer_s2l2a.decode_tokens(tokens)

        # Unstandardize
        reconstruction_unstd = (reconstruction * std[modality] + mean[modality])[0]

    print(f"Sample key: {sample['__key__']}")
    print(f"Reconstruction shape: {reconstruction_unstd.shape}")


#%% md
# ## Visualize Original vs Reconstruction
#%%
def plot_s2_rgb(data, ax=None, title=None):
    """Plot S2 RGB bands (B, G, R = bands 1, 2, 3)"""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    
    # Extract RGB bands and normalize
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    
    rgb = data[[3, 2, 1], :, :]  # R, G, B
    rgb = np.transpose(rgb, (1, 2, 0))
    
    # Normalize to 0-1
    rgb = np.clip(rgb / 3000, 0, 1)
    
    ax.imshow(rgb)
    ax.axis('off')
    if title:
        ax.set_title(title)
    
    return ax

#%% md
# ## Load TerraMind v1.5 Generation Model
#%%
# Load TerraMind v1.5 checkpoint for generation
checkpoint_path = '/Users/jja/Documents/02_EOFM/TerraMind-Pretraining/checkpoints/v1_5/baseline/checkpoint_14.pth'

# Build generation model with multicodebook support
from terratorch.models.backbones.terramind.model.terramind_generation import TerraMindGeneration
from terratorch.models.backbones.terramind.model.terramind_register import tokenizer_dict

model = TerraMindGeneration(
    img_size=224,
    modalities=['S2L2A'],
    # output_modalities=['S1GRD', 'DEM', 'LULC'],
    output_modalities=['S2L2A'],
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
missing_keys, unexpected_keys = model.sampler.model.load_state_dict(checkpoint['model'], strict=False)
print(f"Missing keys: {len(missing_keys)}")

model.eval()
model.to(device)

print(f"TerraMind v1.5 model loaded from: {checkpoint_path}")
print(f"Epoch: {checkpoint.get('epoch', 'unknown')}")
#%% md
# ## Generate All Modalities
#%%
# Run generation and save tokens
tokens_save_path = 'generated_tokens.npy'
with torch.no_grad():
    generated = model(input_standardized, verbose=True, save_tokens_path=tokens_save_path)

print("\nGenerated modalities:")
for mod, value in generated.items():
    print(f"  {mod}: {value.shape}")

print(f"\nGenerated tokens saved to: {tokens_save_path}")
#%% md
# ## Visualize All Results
#%%
# Load and decode the saved tokens from generation process
tok = np.load("/Users/jja/Downloads/tokens_out_dict_v2.npy", allow_pickle=True)

print("\n=== Loaded Generated Tokens from Generation Process ===")
print(f"{mod}: shape={tok.shape}, dtype={tok.dtype}, range=[{tok.min():.0f}, {tok.max():.0f}]")

# Get the tokens for the output modality
# The tokens are saved as (num_steps, B, num_select, num_codebooks) from the generation loop
# We want the final state, so take the last step

# Take the last generated tokens (final step)
# Shape should be (B, num_select, num_codebooks) for the last step
loaded_tokens_final = torch.from_numpy(tok).to(device)
print(f"\nFinal tokens shape: {loaded_tokens_final.shape}")

num_codebooks = 128
image_size = (224, 224)

# Decode the loaded tokens
# The tokens from generation are in shape (B, num_select, num_codebooks)
# We need to reshape to (B, num_codebooks, H, W) for the tokenizer
from einops import rearrange
patch_size = tokenizer_s2l2a.patch_size
h = image_size[0] // patch_size
w = image_size[1] // patch_size

if num_codebooks > 1:
    # Rearrange: (B, num_tokens, num_codebooks) -> (B, num_codebooks, H, W)
    loaded_tokens_reshaped = rearrange(
        loaded_tokens_final, "b (nh nw) c -> b nh nw c", nh=h, nw=w, c=num_codebooks
    )
else:
    # Single codebook: (B, num_tokens) -> (B, H, W)
    loaded_tokens_reshaped = rearrange(
        loaded_tokens_final, "b (nh nw) -> b nh nw", nh=h, nw=w
    )

print(f"Reshaped tokens for decoder: {loaded_tokens_reshaped.shape}")

# Decode tokens using the tokenizer
with torch.no_grad():
    decoded_from_saved_tokens = tokenizer_s2l2a.decode_tokens(loaded_tokens_reshaped)
    decoded_from_saved_tokens_unstd = (decoded_from_saved_tokens * std[modality] + mean[modality])[0]

print(f"Decoded from saved tokens shape: {decoded_from_saved_tokens_unstd.shape}")

# Create comprehensive visualization
n_plots = 3 + len(generated)  # Original + Reconstruction + Decoded Tokens + Generated modalities
fig, axes = plt.subplots(2, (n_plots + 1) // 2, figsize=(6 * ((n_plots + 1) // 2), 12))
axes = axes.flatten()

# Plot original
plot_s2_rgb(data[modality], ax=axes[0], title='Original S2L2A Input')

# Plot tokenizer reconstruction
plot_modality(mod, reconstruction_unstd, ax=axes[1], title='Tokenizer Reconstruction\n(128 codebooks)')

# Plot decoded from saved tokens
plot_modality(mod, decoded_from_saved_tokens_unstd, ax=axes[2], title='Decoded from Saved Tokens\n(Generated)')
# plt.imshow(decoded_from_saved_tokens_unstd.cpu().numpy())

# Plot generated modalities
for i, (mod, value) in enumerate(generated.items(), start=3):
    mod_name = mod.replace('tok_', '').replace('@224', '').upper()
    generated_unstd = (value[0] * std[modality] + mean[modality])
    plot_modality(mod, generated_unstd[0].cpu().numpy(), ax=axes[3], title='Generated S2L2A\n(128 codebooks)')

# Hide unused subplots
for i in range(n_plots, len(axes)):
    axes[i].axis('off')

plt.tight_layout()
plt.savefig('terramind_v1_5_generation_results.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nVisualization complete! Results saved to 'terramind_v1_5_generation_results.png'")
#%% md
# ## Analyze Token Statistics
#%%
# Analyze multicodebook token distribution
print("\n=== Token Statistics ===")
print(f"Token shape: {tokens.shape}")  # (B, num_codebooks, H, W)
print(f"Number of codebooks: {tokens.shape[1]}")
print(f"Spatial resolution: {tokens.shape[2]} x {tokens.shape[3]}")
print(f"\nToken value range: [{tokens.min().item():.0f}, {tokens.max().item():.0f}]")
print(f"Expected range: [0, 7] (8 FSQ levels)")

# Plot token distribution per codebook
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()
