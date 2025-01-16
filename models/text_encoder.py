import torch
import torch.nn as nn
from transformers import RobertaModel


class TextEncoder(nn.Module):
    """
    A wrapper around RoBERTa to extract the hidden state of 
    the [CLS] token (token_idx=0) for text embeddings.
    """
    def __init__(self):
        super().__init__()
        self.model = RobertaModel.from_pretrained("roberta-base")
        self.target_token_idx = 0  # Using the "<s>" token as the sentence embedding

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)

        Returns:
            torch.Tensor of shape (batch_size, hidden_dim) corresponding 
            to the embedding of the target token.
        """
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[:, self.target_token_idx, :]
