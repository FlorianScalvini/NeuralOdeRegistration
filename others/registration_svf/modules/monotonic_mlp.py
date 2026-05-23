import torch
import torch.nn as nn
import torch.nn.functional as F



class MonotonicMLP(nn.Module):

    def __init__(self, hidden=32, n_points=1000):
        super().__init__()
        self.n_points = n_points
        self.fc1 = nn.Linear(1, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 1)

    def forward(self, x):
        # x: [B,1] values in [0,1]
        # Sample g(x) on a fine grid
        grid = torch.linspace(0, 1, self.n_points).unsqueeze(1).to(x.device)  # [n_points,1]
        h = F.relu(self.fc1(grid))
        h = F.relu(self.fc2(h))
        g = torch.exp(self.fc3(h)) + 1e-6  # small epsilon to avoid zero

        # 3. Compute cumulative integral (trapezoid rule)
        dx = 1.0 / (self.n_points - 1)
        cum_g = torch.cumsum(g[:-1] + g[1:], dim=0) * (dx / 2)
        cum_g = torch.cat([torch.zeros(1, 1).to(x.device), cum_g], dim=0)  # prepend 0
        cum_g = cum_g / cum_g[-1]  # normalize so f(1)=1

        # Map x to grid indices (scaled)
        x_flat = x.view(-1)
        scaled = x_flat * (self.n_points - 1)
        idx_lower = scaled.floor().long().clamp(0, self.n_points - 2)
        idx_upper = idx_lower + 1
        alpha = (scaled - idx_lower.float())

        # Linear interpolation
        f_flat = (1 - alpha) * cum_g[idx_lower] + alpha * cum_g[idx_upper]
        f = f_flat.view_as(x)
        return f
