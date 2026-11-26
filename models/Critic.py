# ==============================================================================
# File source: EfficientPlace-ERL-Para_Element/models/critic.py
# Purpose: Paper-aligned final ERLPlace code snapshot.
# Note: Copied without code changes; only this source header was added.
# ==============================================================================
import torch
import torch.nn as nn
from torch import Tensor
from typing import Any

class Critic(nn.Module):
    def __init__(self, num_macros_to_place: int, grid: int):
        super(Critic, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=int(grid / 128), stride=int(grid / 128)),
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
        )
        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
        )
        self.pool_2 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.conv_3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
        )
        self.pool_3 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.conv_4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(4 * 4 * 32 + 32, 512),
            nn.Tanh(),
            nn.Linear(512, 512),
        )
        self.time_embedding = nn.Embedding(num_macros_to_place + 1, 32)
        self.fc2 = nn.Sequential(
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, 1),
        )
        self.time_value = nn.Embedding(num_macros_to_place + 1, 1)
    def __call__(self, *args: Any, **kwargs: Any) -> Tensor:
        return self.forward(*args, **kwargs)
    def forward(self, s: Tensor, t: Tensor):
        feature_1 = self.conv_1(s)
        feature_1_pool = self.pool_1(feature_1)
        feature_2 = self.conv_2(feature_1_pool)
        feature_2_pool = self.pool_2(feature_2)
        feature_3 = self.conv_3(feature_2_pool)
        feature_3_pool = self.pool_3(feature_3)
        feature_4 = self.conv_4(feature_3_pool)
        time_embedding = self.time_embedding(t)
        feature = torch.cat([torch.reshape(feature_4, [-1, 4 * 4 * 32]), time_embedding], dim=-1)
        feature = self.fc1(feature)
        value = self.fc2(feature) + self.time_value(t)
        return value

if __name__ == "__main__":
    critic = Critic(128, 256)
    for name, param in critic.named_parameters():
        print(name, param.size())
    

