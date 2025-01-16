import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCE(nn.Module):
    """
    A simple InfoNCE loss used in CLIP-like contrastive training.
    """
    def __init__(self, temp: float = 1.0) -> None:
        super().__init__()
        self.temperature = temp

    def forward(self, text_embeddings: torch.Tensor, audio_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute the pairwise cosine similarities of text vs. audio embeddings 
        and apply a contrastive loss to align matching text/audio pairs.

        Args:
            text_embeddings (torch.Tensor): Shape (batch_size, dim)
            audio_embeddings (torch.Tensor): Shape (batch_size, dim)

        Returns:
            torch.Tensor: Scalar loss.
        """
        logits = (text_embeddings @ audio_embeddings.T) / self.temperature
        audio_similarity = audio_embeddings @ audio_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T
        targets = F.softmax((audio_similarity + texts_similarity) / 2 * self.temperature, dim=-1)

        loss_texts = self.cross_entropy(logits, targets)
        loss_audio = self.cross_entropy(logits.T, targets.T)
        loss = loss_texts + loss_audio
        return loss.mean()

    @staticmethod
    def cross_entropy(preds: torch.Tensor, targets: torch.Tensor, reduction: str = 'none') -> torch.Tensor:
        """
        Custom cross entropy that uses soft targets.
        """
        log_softmax_vals = F.log_softmax(preds, dim=-1)
        loss = (-targets * log_softmax_vals).sum(dim=1)
        if reduction == "mean":
            return loss.mean()
        return loss
