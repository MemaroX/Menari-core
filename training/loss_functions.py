import torch
import torch.nn.functional as F

class CrossEntropyLoss:
    """A basic implementation of cross-entropy loss using PyTorch."""
    def calculate_loss(self, predictions, targets):
        """
        Calculates cross-entropy loss.
        Args:
            predictions (torch.Tensor): Logits from the model (batch_size, seq_len, vocab_size).
            targets (torch.Tensor): True token IDs (batch_size, seq_len).
        Returns:
            torch.Tensor: The average loss.
        """
        # Reshape predictions to (batch_size * seq_len, vocab_size)
        predictions = predictions.view(-1, predictions.size(-1))
        # Reshape targets to (batch_size * seq_len)
        targets = targets.view(-1)
        return F.cross_entropy(predictions, targets)
