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

from .decoder_embeddings import ImageTokenDecoderEmbedding, SequenceDecoderEmbedding
from .encoder_embeddings import ImageEncoderEmbedding, ImageTokenEncoderEmbedding, SequenceEncoderEmbedding
from terratorch.models.backbones.terramind.utils import generate_uint15_hash

MODALITY_INFO = {
    "sen1grd@264": {
        "input_size": 264,
        "patch_size": 16,
        "encoder_embedding": partial(ImageEncoderEmbedding, num_channels=2),
        "decoder_embedding": partial(ImageEncoderEmbedding, num_channels=2),
        # 'decoder_embedding': None,
        "min_tokens": 0,
        "max_tokens": None,  # Will be set to 196
        "type": "img",
        "num_channels": 2,
        "id": generate_uint15_hash("sen1grd@264"),
        "path": "S1GRD/",
        "data_range": 51,
    },
    "sen1rtc@264": {
        "input_size": 264,
        "patch_size": 16,
        "encoder_embedding": partial(ImageEncoderEmbedding, num_channels=2),
        "decoder_embedding": partial(ImageEncoderEmbedding, num_channels=2),
        # 'decoder_embedding': None,
        "min_tokens": 0,
        "max_tokens": None,  # Will be set to 196
        "type": "img",
        "num_channels": 2,
        "id": generate_uint15_hash("sen1rtc@264"),
        "path": "S1RTC/",
        "data_range": 51,
    },
    "sen2l2a@264": {
        "input_size": 264,
        "patch_size": 16,
        "encoder_embedding": partial(ImageEncoderEmbedding, num_channels=12),
        "decoder_embedding": partial(ImageEncoderEmbedding, num_channels=12),
        # 'decoder_embedding': None,
        "min_tokens": 0,
        "max_tokens": None,  # Will be set to 196
        "type": "img",
        "num_channels": 12,
        "id": generate_uint15_hash("sen2l2a@264"),
        "path": "S2L2A/",
        "data_range": 10000,
    },
    "sen2l1c@264": {
        "input_size": 264,
        "patch_size": 16,
        "encoder_embedding": partial(ImageEncoderEmbedding, num_channels=12),
        "decoder_embedding": partial(ImageEncoderEmbedding, num_channels=12),
        # 'decoder_embedding': None,
        "min_tokens": 0,
        "max_tokens": None,  # Will be set to 196
        "type": "img",
        "num_channels": 13,
        "id": generate_uint15_hash("sen2l1c@264"),
        "path": "S2L1C/",
        "data_range": 10000,
    },
    "sen2rgb@264": {
        "input_size": 264,
        "patch_size": 16,
        "encoder_embedding": partial(ImageEncoderEmbedding, num_channels=3),
        "decoder_embedding": partial(ImageEncoderEmbedding, num_channels=3),
        # 'decoder_embedding': None,
        "min_tokens": 0,
        "max_tokens": None,  # Will be set to 196
        "type": "img",
        "num_channels": 3,
        "id": generate_uint15_hash("sen2rgb@264"),
        "path": "S2RGB/",
        "data_range": 255,
    },
    "lulc@264": {
        "input_size": 264,
        "patch_size": 16,
        "encoder_embedding": partial(ImageEncoderEmbedding, num_channels=9),
        "decoder_embedding": partial(ImageEncoderEmbedding, num_channels=9),
        "min_tokens": 0,
        "max_tokens": None,  # Will be set to 196
        "type": "img",
        "num_channels": 10,
        "id": generate_uint15_hash("lulc@264"),
        "path": "LULC/",
    },
    "dem@264": {
        "input_size": 264,
        "patch_size": 16,
        "encoder_embedding": partial(ImageEncoderEmbedding, num_channels=1),
        "decoder_embedding": partial(ImageEncoderEmbedding, num_channels=1),
        "min_tokens": 0,
        "max_tokens": None,  # Will be set to 196
        "type": "img",
        "num_channels": 1,
        "id": generate_uint15_hash("dem@264"),
        "path": "DEM/",
        "data_range": 8000,
    },
    "ndvi@264": {
        "input_size": 264,
        "patch_size": 16,
        "encoder_embedding": partial(ImageEncoderEmbedding, num_channels=1),
        "decoder_embedding": partial(ImageEncoderEmbedding, num_channels=1),
        "min_tokens": 0,
        "max_tokens": None,  # Will be set to 196
        "type": "img",
        "num_channels": 1,
        "id": generate_uint15_hash("ndvi@264"),
        "path": "NDVI/",
        "data_range": 2,
    },
    "naip@512": {
        "input_size": 512,
        "patch_size": 16,
        "encoder_embedding": partial(ImageEncoderEmbedding, num_channels=4),
        "decoder_embedding": None,
        "min_tokens": 0,
        "max_tokens": None,  # Will be set to 196
        "type": "img",
        "num_channels": 4,
        "id": generate_uint15_hash("naip@512"),
        "path": "NAIP",
        "data_range": 255,
    },
    "untok_sen2l2a@224": {  # untokenized version
        "input_size": 224,
        "patch_size": 16,
        "encoder_embedding": partial(ImageEncoderEmbedding, num_channels=12),
        "decoder_embedding": None,
        "min_tokens": 0,
        "max_tokens": None,  # Will be set to 196
        "type": "img",
        "num_channels": 12,
        "id": generate_uint15_hash("untok_sen2l2a@224"),
        "path": "S2L2A",
    },
    "untok_sen2l1c@224": {  # untokenized version
        "input_size": 224,
        "patch_size": 16,
        "encoder_embedding": partial(ImageEncoderEmbedding, num_channels=13),
        "decoder_embedding": None,
        "min_tokens": 0,
        "max_tokens": None,  # Will be set to 196
        "type": "img",
        "num_channels": 13,
        "id": generate_uint15_hash("untok_sen2l1c@224"),
        "path": "S2L1C",
    },
    "untok_sen2rgb@224": {  # untokenized version
        "input_size": 224,
        "patch_size": 16,
        "encoder_embedding": partial(ImageEncoderEmbedding, num_channels=3),
        "decoder_embedding": None,
        "min_tokens": 0,
        "max_tokens": None,  # Will be set to 196
        "type": "img",
        "num_channels": 3,
        "id": generate_uint15_hash("untok_sen2rgb@224"),
        "path": "S2RGB",
    },
    "untok_sen1grd@224": {  # untokenized version
        "input_size": 224,
        "patch_size": 16,
        "encoder_embedding": partial(ImageEncoderEmbedding, num_channels=2),
        "decoder_embedding": None,
        "min_tokens": 0,
        "max_tokens": None,  # Will be set to 196
        "type": "img",
        "num_channels": 2,
        "id": generate_uint15_hash("untok_sen1grd@224"),
        "path": "S1GRD",
    },
    "untok_sen1rtc@224": {  # untokenized version
        "input_size": 224,
        "patch_size": 16,
        "encoder_embedding": partial(ImageEncoderEmbedding, num_channels=2),
        "decoder_embedding": None,
        "min_tokens": 0,
        "max_tokens": None,  # Will be set to 196
        "type": "img",
        "num_channels": 2,
        "id": generate_uint15_hash("untok_sen1rtc@224"),
        "path": "S1RTC",
    },
    "tok_sen1grd@224": {
        "input_size": 224,
        "patch_size": 16,
        "vocab_size": 8,
        "encoder_embedding": partial(ImageTokenEncoderEmbedding, vocab_size=8),
        "decoder_embedding": partial(ImageTokenDecoderEmbedding, vocab_size=8),
        "min_tokens": 0,
        "max_tokens": None,  # Will be set to 196
        "type": "img",
        "id": generate_uint15_hash("tok_sen1grd@224"),
        "pretokenized": True,
        "path": "S1GRD_tokens_v1_5",
    },
    "tok_sen1rtc@224": {
        "input_size": 224,
        "patch_size": 16,
        "vocab_size": 8,
        "encoder_embedding": partial(ImageTokenEncoderEmbedding, vocab_size=8),
        "decoder_embedding": partial(ImageTokenDecoderEmbedding, vocab_size=8),
        "min_tokens": 0,
        "max_tokens": None,  # Will be set to 196
        "type": "img",
        "id": generate_uint15_hash("tok_sen1rtc@224"),
        "pretokenized": True,
        "path": "S1RTC_tokens_v1_5",
    },
    "tok_sen2l2a@224": {
        "input_size": 224,
        "patch_size": 16,
        "vocab_size": 8,
        "encoder_embedding": partial(ImageTokenEncoderEmbedding, vocab_size=8),
        "decoder_embedding": partial(ImageTokenDecoderEmbedding, vocab_size=8),
        "min_tokens": 0,
        "max_tokens": None,  # Will be set to 196
        "type": "img",
        "id": generate_uint15_hash("tok_sen2l2a@224"),
        "pretokenized": True,
        "path": "S2L2A_tokens_v1_5",
    },
    "tok_sen2l1c@224": {
        "input_size": 224,
        "patch_size": 16,
        "vocab_size": 8,
        "encoder_embedding": partial(ImageTokenEncoderEmbedding, vocab_size=8),
        "decoder_embedding": partial(ImageTokenDecoderEmbedding, vocab_size=8),
        "min_tokens": 0,
        "max_tokens": None,  # Will be set to 196
        "type": "img",
        "id": generate_uint15_hash("tok_sen2l1c@224"),
        "pretokenized": True,
        "path": "S2L1C_tokens_v1_5",
    },
    "tok_sen2rgb@224": {
        "input_size": 224,
        "patch_size": 16,
        "vocab_size": 8,
        "encoder_embedding": partial(ImageTokenEncoderEmbedding, vocab_size=8),
        "decoder_embedding": partial(ImageTokenDecoderEmbedding, vocab_size=8),
        "min_tokens": 0,
        "max_tokens": None,  # Will be set to 196
        "type": "img",
        "id": generate_uint15_hash("tok_sen2rgb@224"),
        "pretokenized": True,
        "path": "S2RGB_tokens_v1_5",
    },
    "tok_lulc@224": {
        "input_size": 224,
        "patch_size": 16,
        "vocab_size": 8,
        "encoder_embedding": partial(ImageTokenEncoderEmbedding, vocab_size=8),
        "decoder_embedding": partial(ImageTokenDecoderEmbedding, vocab_size=8),
        "min_tokens": 0,
        "max_tokens": None,  # Will be set to 196
        "type": "img",
        "id": generate_uint15_hash("tok_lulc@224"),
        "pretokenized": True,
        "path": "LULC_tokens_v1_5",
    },
    "untok_dem@224": {  # untokenized version
        "input_size": 224,
        "patch_size": 16,
        "encoder_embedding": partial(ImageEncoderEmbedding, num_channels=1),
        "decoder_embedding": None,
        "min_tokens": 0,
        "max_tokens": None,  # Will be set to 196
        "type": "img",
        "num_channels": 1,
        "id": generate_uint15_hash("untok_dem@224"),
        "path": "DEM",
    },
    "tok_dem@224": {
        "input_size": 224,
        "patch_size": 16,
        "vocab_size": 8,
        "encoder_embedding": partial(ImageTokenEncoderEmbedding, vocab_size=8),
        "decoder_embedding": partial(ImageTokenDecoderEmbedding, vocab_size=8),
        "min_tokens": 0,
        "max_tokens": None,  # Will be set to 196
        "type": "img",
        "id": generate_uint15_hash("tok_dem@224"),
        "pretokenized": True,
        "path": "DEM_tokens_v1_5",
    },
    "tok_ndvi@224": {
        "input_size": 224,
        "patch_size": 16,
        "vocab_size": 8,
        "encoder_embedding": partial(ImageTokenEncoderEmbedding, vocab_size=8),
        "decoder_embedding": partial(ImageTokenDecoderEmbedding, vocab_size=8),
        "min_tokens": 0,
        "max_tokens": None,  # Will be set to 196
        "type": "img",
        "id": generate_uint15_hash("tok_ndvi@224"),
        "pretokenized": True,
        "path": "NDVI_tokens_v1_5",
    },
    ### Natural image/text domains
    "caption": {
        "vocab_size": 30_000,
        "encoder_embedding": partial(
            SequenceEncoderEmbedding, vocab_size=30_000, max_length=256, padding_idx=0
        ),
        "decoder_embedding": partial(
            SequenceDecoderEmbedding, vocab_size=30_000, max_length=256, padding_idx=0
        ),
        "min_tokens": 0,
        "max_tokens": 256,
        "type": "seq",
        "id": generate_uint15_hash("caption"),
        "path": "captions_txt",
    },
    "coords": {
        "vocab_size": 30_000,
        "encoder_embedding": partial(
            SequenceEncoderEmbedding, vocab_size=30_000, max_length=256, padding_idx=0
        ),
        "decoder_embedding": partial(
            SequenceDecoderEmbedding, vocab_size=30_000, max_length=256, padding_idx=0
        ),
        "min_tokens": 0,
        "max_tokens": 256,
        "type": "seq",
        "id": generate_uint15_hash("coords"),
        "path": "coords",
    },
}
