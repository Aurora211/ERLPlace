import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Categorical
from typing import Any, Tuple

class Actor(nn.Module):
    def __init__(self, number_of_macros_to_place: int, grid: int):
        super(Actor, self).__init__()
        self.grid = grid
        self.conv_1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=int(grid / 128), stride=int(grid / 128)),
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.pool_2 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.conv_3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.pool_3 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.conv_4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(4 * 4 * 32 + 32, 512),
            nn.ReLU(),
            nn.Linear(512, 4 * 4 * 32),
            nn.ReLU(),
        )
        self.up_5 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.conv_5 = nn.Sequential(
            nn.Conv2d(64, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.up_6 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.conv_6 = nn.Sequential(
            nn.Conv2d(32, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.up_7 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_7 = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.conv_8 = nn.Sequential(
            nn.Upsample(scale_factor=int(grid / 128), mode='bilinear', align_corners=True),
            nn.Conv2d(8, 1, kernel_size=1, stride=1),
        )
        self.time_embedding = nn.Embedding(number_of_macros_to_place, 32)
    def forward(self, s, t):
        feature_1 = self.conv_1(s)
        feature_1_pool = self.pool_1(feature_1)
        feature_2 = self.conv_2(feature_1_pool)
        feature_2_pool = self.pool_2(feature_2)
        feature_3 = self.conv_3(feature_2_pool)
        feature_3_pool = self.pool_3(feature_3)
        feature_4 = self.conv_4(feature_3_pool)
        time_embedding = self.time_embedding(t)
        feature = torch.cat([torch.reshape(feature_4, [-1, 4 * 4 * 32]), time_embedding], dim=-1)
        feature = torch.reshape(self.fc(feature), [-1, 32, 4, 4])
        feature_5 = self.up_5(feature)
        feature_5 = self.conv_5(torch.cat((feature_3, feature_5), dim=1))
        feature_6 = self.up_6(feature_5)
        feature_6 = self.conv_6(torch.cat((feature_2, feature_6), dim=1))
        feature_7 = self.up_7(feature_6)
        feature_7 = self.conv_7(torch.cat((feature_1, feature_7), dim=1))
        feature_8 = self.conv_8(feature_7)
        return torch.reshape(feature_8, [-1, self.grid * self.grid])
    def get_distribution(self, logits: Tensor, position_mask: Tensor, wire_mask: Tensor):
        mask = torch.where(position_mask > 0.5, torch.inf, wire_mask).reshape([-1, self.grid * self.grid])
        wire_min = mask.min(dim=-1, keepdim=True)[0]
        greedy_mask = torch.where(mask == wire_min, 0.0, - 1e10).view([-1, self.grid * self.grid])
        action_mask = torch.reshape(position_mask >= 0.5, [-1, self.grid * self.grid])
        pi_distr = torch.where(action_mask, - 1e10, logits)
        greedy_distr = pi_distr + greedy_mask
        pi_distr = torch.softmax(pi_distr, dim=-1)
        pi_distr_no_mask = torch.softmax(logits, dim=-1)
        greedy_distr = torch.softmax(greedy_distr, dim=-1)
        action_distr = greedy_distr
        distr = Categorical(action_distr)
        return distr, pi_distr, action_distr, pi_distr_no_mask
    def __call__(self, s, t, *args: Any, **kwargs: Any) -> Tuple[Categorical, Tensor, Tensor, Tensor]:
        logits = self.forward(s, t)
        distr, pi_distr, action_distr, pi_distr_nomask = self.get_distribution(logits, position_mask=kwargs['position_mask'], wire_mask=kwargs['wire_mask'])
        return distr, pi_distr, action_distr, pi_distr_nomask

if __name__ == "__main__":
    actor = Actor(128, 256)
    for name, param in actor.named_parameters():
        print(name, param.size())


