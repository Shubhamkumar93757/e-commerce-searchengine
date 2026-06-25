import torch
import torch.nn as nn
import torch.nn.functional as F

class SharedSpaceProjector(nn.Module):
    def __init__(self, embedding_dim=512, hidden_dim=256):
        super().__init__()
        self.vision_proj= nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )

        self.text_proj= nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,embedding_dim)

        )

    def forward(self, vision_emb, text_emb):

        vision_proj = F.normalize(self.vision_proj(vision_emb), p=2, dim=1)
        text_proj = F.normalize(self.text_proj(text_emb), p=2,dim=1)
        return vision_proj, text_proj

class TripleLoss(nn.Module):
        
    def __int__(self, margin=0.5):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):

        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)
        loss= F.relu(pos_dist- neg_dist+ self.margin).mean()
        return loss        