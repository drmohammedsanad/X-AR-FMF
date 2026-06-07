import torch
import torch.nn as nn

class AdversarialMF(nn.Module):
    """
    Adversarial fairness model
    Learns representation + predicts group
    """

    def __init__(self, num_users, num_items, k=20):
        super().__init__()

        self.user_emb = nn.Embedding(num_users, k)
        self.item_emb = nn.Embedding(num_items, k)

        # Rating prediction
        self.predictor = nn.Linear(k, 1)

        # Group prediction (fairness adversary)
        self.adversary = nn.Linear(k, 2)

    def forward(self, user, item):
        u = self.user_emb(user)
        i = self.item_emb(item)

        z = u * i  # element-wise interaction

        rating = self.predictor(z)
        group_pred = self.adversary(z)

        return rating, group_pred