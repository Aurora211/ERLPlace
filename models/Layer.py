from typing import List

# Size on
# Macro: 128
# Grid: 256

ACTOR_LAYERS: List = [
    {"prefix": "conv_1.1",       "size": {"weight": [8, 3, 3, 3],   "bias": [8]}},
    {"prefix": "conv_2.0",       "size": {"weight": [16, 8, 3, 3],  "bias": [16]}},
    {"prefix": "conv_3.0",       "size": {"weight": [32, 16, 3, 3], "bias": [32]}},
    {"prefix": "conv_4.0",       "size": {"weight": [32, 32, 3, 3], "bias": [32]}},
    {"prefix": "fc.0",           "size": {"weight": [512, 544],     "bias": [512]}},
    {"prefix": "fc.2",           "size": {"weight": [512, 512],     "bias": [512]}},
    {"prefix": "conv_5.0",       "size": {"weight": [16, 64, 3, 3], "bias": [16]}},
    {"prefix": "conv_6.0",       "size": {"weight": [8, 32, 3, 3],  "bias": [8]}},
    {"prefix": "conv_7.0",       "size": {"weight": [8, 16, 3, 3],  "bias": [8]}},
    {"prefix": "conv_8.1",       "size": {"weight": [1, 8, 1, 1],   "bias": [1]}},
    {"prefix": "time_embedding", "size": {"weight": [128, 32]}},
]
CRITIC_LAYERS: List = [
    {"prefix": "conv_1.1",       "size": {"weight": [8, 3, 3, 3],   "bias": [8]}},
    {"prefix": "conv_2.0",       "size": {"weight": [16, 8, 3, 3],  "bias": [16]}},
    {"prefix": "conv_3.0",       "size": {"weight": [32, 16, 3, 3], "bias": [32]}},
    {"prefix": "conv_4.0",       "size": {"weight": [32, 32, 3, 3], "bias": [32]}},
    {"prefix": "fc1.0",          "size": {"weight": [512, 544],     "bias": [512]}},
    {"prefix": "fc1.2",          "size": {"weight": [512, 512],     "bias": [512]}},
    {"prefix": "time_embedding", "size": {"weight": [128, 32]}},
    {"prefix": "fc2.0",          "size": {"weight": [512, 512],     "bias": [512]}},
    {"prefix": "fc2.2",          "size": {"weight": [1, 512],       "bias": [1]}},
    {"prefix": "time_value",     "size": {"weight": [129, 1]}},
]


