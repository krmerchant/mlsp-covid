import torch
import torch.nn as nn

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # Calculate Euclidean distance between the two embeddings
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        # Compute contrastive loss
        loss = torch.mean((label) * torch.pow(euclidean_distance, 2) +
                          (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0), 2))

        return loss
