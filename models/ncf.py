import torch
import torch.nn as nn

class NCF(nn.Module):
    """
    Neural Collaborative Filtering
    Lightweight version for evaluation
    """

    def __init__(self, num_users, num_items, k=16):
        super().__init__()

        self.user_emb = nn.Embedding(num_users, k)
        self.item_emb = nn.Embedding(num_items, k)

        self.mlp = nn.Sequential(
            nn.Linear(2 * k, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, user, item):
        u = self.user_emb(user)
        i = self.item_emb(item)

        x = torch.cat([u, i], dim=1)
        return self.mlp(x)