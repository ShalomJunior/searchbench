import torch
import torch.nn.functional as F


def infonce_loss(
    pos_scores: torch.Tensor, neg_scores: torch.Tensor, temperature: float = 1.0
) -> torch.Tensor:
    """
    Computes the InfoNCE (Noise-Contrastive Estimation) loss for a batch of ranking scores.
    This effectively frames ranking as a multiple-choice classification problem.

    Args:
        pos_scores: Tensor of shape (batch_size, 1) containing scores for positive (relevant) documents.
        neg_scores: Tensor of shape (batch_size, num_negatives) containing scores for negative documents.
        temperature: Float hyperparameter controlling the sharpness of the distribution.

    Returns:
        Scalar tensor containing the computed loss.
    """
    # 1. Concatenate positive and negative scores
    # Shape becomes: (batch_size, 1 + num_negatives)
    logits = torch.cat([pos_scores, neg_scores], dim=1)

    # 2. Apply temperature scaling
    # This dictates how severely the model penalizes the hardest negatives.
    logits = logits / temperature

    # 3. The correct "class" for each query is always index 0 (the positive document)
    # Since we concatenated the positive scores at the 0th index for every row.
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)

    # 4. Compute standard Cross-Entropy
    # Mathematically, Softmax Cross-Entropy is identical to the InfoNCE formulation.
    return F.cross_entropy(logits, labels)
