"""
Complete GPT-2 language model.
"""

import torch
import torch.nn as nn
from .lookup_nn import LookupNN
from .decoder import GPT2Decoder


class GPT2(nn.Module):
    """
    Complete GPT-2 language model.

    This combines token embeddings, positional embeddings, decoder blocks,
    and a language model head to predict next tokens.

    Architecture:
        1. Token embeddings (vocab_size -> embedding_dim)
        2. Learned positional embeddings (max_seq_len -> embedding_dim)
        3. Decoder blocks (12 layers with self-attention and FFN)
        4. Language model head (embedding_dim -> vocab_size)
        5. Softmax to get token probabilities

    Default configuration matches GPT-2 Small:
        - 12 layers
        - 768 embedding dimension
        - 12 attention heads
        - 3072 FFN hidden dimension (4x)
        - 50257 vocabulary size

    Args:
        vocab_size: Size of vocabulary
        embedding_dim: Dimension of embeddings (default: 768)
        num_layers: Number of decoder layers (default: 12)
        num_heads: Number of attention heads (default: 12)
        max_seq_len: Maximum sequence length (default: 1024)
        dropout: Dropout probability (default: 0.1)
        ffn_hidden_dim: FFN hidden dimension (default: 4 * embedding_dim)
    """

    def __init__(
        self,
        vocab_size,
        embedding_dim=768,
        num_layers=12,
        num_heads=12,
        max_seq_len=1024,
        dropout=0.1,
        ffn_hidden_dim=None
    ):
        super(GPT2, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len

        # Token embeddings
        self.token_embedding = LookupNN(vocab_size, embedding_dim)
        self.token_embedding.create()

        # Learned positional embeddings (GPT-2 style)
        self.position_embedding = nn.Embedding(max_seq_len, embedding_dim)

        # Dropout after embeddings
        self.dropout = nn.Dropout(dropout)

        # Decoder blocks
        self.decoder = GPT2Decoder(
            num_layers=num_layers,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            ffn_hidden_dim=ffn_hidden_dim,
            dropout=dropout,
            use_rope=False,  # GPT-2 uses learned positional embeddings
            max_seq_len=max_seq_len
        )

        # Language model head (projects to vocabulary)
        self.lm_head = nn.Linear(embedding_dim, vocab_size, bias=False)

        # Tie weights between token embeddings and LM head (standard practice)
        self.lm_head.weight = self.token_embedding.embedding.weight

    def forward(self, token_ids, return_logits=False):
        """
        Forward pass through the model.

        Args:
            token_ids: Input token IDs of shape (batch_size, seq_len)
            return_logits: If True, return logits instead of probabilities

        Returns:
            If return_logits=False: Token probabilities of shape (batch_size, seq_len, vocab_size)
            If return_logits=True: Logits of shape (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = token_ids.shape

        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds maximum {self.max_seq_len}")

        # Get token embeddings
        token_embeds = self.token_embedding(token_ids)  # (batch, seq_len, embedding_dim)

        # Get positional embeddings
        positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(0)  # (1, seq_len)
        position_embeds = self.position_embedding(positions)  # (1, seq_len, embedding_dim)

        # Combine embeddings
        x = token_embeds + position_embeds
        x = self.dropout(x)

        # Pass through decoder
        x = self.decoder(x)  # (batch, seq_len, embedding_dim)

        # Project to vocabulary
        logits = self.lm_head(x)  # (batch, seq_len, vocab_size)

        if return_logits:
            return logits

        # Apply softmax to get probabilities
        probs = torch.softmax(logits, dim=-1)  # (batch, seq_len, vocab_size)

        return probs

    def predict_next_token(self, token_ids, top_k=None, temperature=1.0):
        """
        Predict the next token given input token IDs.

        Args:
            token_ids: Input token IDs of shape (batch_size, seq_len)
            top_k: If specified, only consider top k most likely tokens
            temperature: Sampling temperature (higher = more random)

        Returns:
            Dictionary with:
                - 'token_id': Predicted token ID (batch_size,)
                - 'probability': Probability of predicted token (batch_size,)
                - 'top_tokens': Top k token IDs (batch_size, top_k) if top_k specified
                - 'top_probs': Top k probabilities (batch_size, top_k) if top_k specified
        """
        self.eval()
        with torch.no_grad():
            # Get probabilities for next token
            logits = self.forward(token_ids, return_logits=True)
            next_token_logits = logits[:, -1, :]  # (batch, vocab_size)

            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            # Get probabilities
            probs = torch.softmax(next_token_logits, dim=-1)  # (batch, vocab_size)

            result = {}

            if top_k is not None:
                # Get top k tokens and probabilities
                top_probs, top_indices = torch.topk(probs, top_k, dim=-1)
                result['top_tokens'] = top_indices
                result['top_probs'] = top_probs

                # Predicted token is the most likely
                result['token_id'] = top_indices[:, 0]
                result['probability'] = top_probs[:, 0]
            else:
                # Get most likely token
                token_prob, token_id = torch.max(probs, dim=-1)
                result['token_id'] = token_id
                result['probability'] = token_prob

            return result

    def get_token_predictions(self, token_ids, position=-1, top_k=10):
        """
        Get top k token predictions and their probabilities at a specific position.

        Args:
            token_ids: Input token IDs of shape (batch_size, seq_len)
            position: Position to get predictions for (default: -1, last position)
            top_k: Number of top predictions to return (default: 10)

        Returns:
            List of tuples (token_id, probability) for each batch item.
            Each batch item contains top_k predictions.
            Shape: batch_size x [(token_id, probability), ...]
        """
        self.eval()
        with torch.no_grad():
            # Get probabilities
            probs = self.forward(token_ids, return_logits=False)  # (batch, seq_len, vocab_size)

            # Get predictions at specified position
            position_probs = probs[:, position, :]  # (batch, vocab_size)

            # Get top k for each batch item
            top_probs, top_indices = torch.topk(position_probs, top_k, dim=-1)

            # Convert to list of tuples
            batch_size = token_ids.shape[0]
            results = []

            for batch_idx in range(batch_size):
                predictions = [
                    (top_indices[batch_idx, i].item(), top_probs[batch_idx, i].item())
                    for i in range(top_k)
                ]
                results.append(predictions)

            return results

    def generate(self, token_ids, max_new_tokens=50, temperature=1.0, top_k=None, stop_token_id=None):
        """
        Generate new tokens autoregressively.

        Args:
            token_ids: Input token IDs of shape (batch_size, seq_len)
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            top_k: If specified, sample from top k tokens
            stop_token_id: If specified, stop when this token is generated

        Returns:
            Generated token IDs of shape (batch_size, seq_len + num_generated)
        """
        self.eval()
        generated = token_ids.clone()

        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Truncate to max_seq_len if needed
                if generated.shape[1] > self.max_seq_len:
                    generated = generated[:, -self.max_seq_len:]

                # Get next token prediction
                result = self.predict_next_token(generated, top_k=top_k, temperature=temperature)
                next_token = result['token_id'].unsqueeze(1)  # (batch, 1)

                # Append to sequence
                generated = torch.cat([generated, next_token], dim=1)

                # Check for stop token
                if stop_token_id is not None and (next_token == stop_token_id).all():
                    break

        return generated

    def save(self, filepath):
        """Save the model weights."""
        torch.save(self.state_dict(), filepath)

    def load(self, filepath):
        """Load the model weights."""
        self.load_state_dict(torch.load(filepath))

    def count_parameters(self):
        """Count the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
