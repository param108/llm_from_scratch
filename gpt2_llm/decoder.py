"""
GPT-2 style transformer decoder block with attention, FFN, and residual connections.
"""

import torch
import torch.nn as nn
from .attention import CausalQKVAttention


class FeedForward(nn.Module):
    """
    Feed-forward network (MLP) used in transformer blocks.

    This implements the position-wise feed-forward network:
    FFN(x) = max(0, xW1 + b1)W2 + b2

    GPT-2 typically uses a 4x expansion factor for the hidden dimension.

    Args:
        embedding_dim: Dimension of input embeddings
        hidden_dim: Dimension of hidden layer (default: 4 * embedding_dim)
        dropout: Dropout probability (default: 0.1)
    """

    def __init__(self, embedding_dim, hidden_dim=None, dropout=0.1):
        super(FeedForward, self).__init__()

        if hidden_dim is None:
            hidden_dim = 4 * embedding_dim

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # Two linear transformations with GELU activation
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        """
        Forward pass through the feed-forward network.

        Args:
            x: Input tensor of shape (batch_size, seq_len, embedding_dim)

        Returns:
            Output tensor of shape (batch_size, seq_len, embedding_dim)
        """
        # First linear transformation + activation
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)

        # Second linear transformation
        x = self.fc2(x)
        x = self.dropout(x)

        return x


class GPT2DecoderBlock(nn.Module):
    """
    Single GPT-2 transformer decoder block.

    This implements the full transformer block with:
    1. Layer normalization before attention (Pre-LN)
    2. Multi-head causal self-attention with residual connection
    3. Layer normalization before FFN
    4. Feed-forward network with residual connection

    The structure follows:
        x = x + attention(LayerNorm(x))
        x = x + ffn(LayerNorm(x))

    Args:
        embedding_dim: Dimension of input embeddings
        num_heads: Number of attention heads
        ffn_hidden_dim: Hidden dimension for FFN (default: 4 * embedding_dim)
        dropout: Dropout probability (default: 0.1)
        use_rope: Whether to use Rotary Position Embeddings (default: False)
        rope_max_seq_len: Maximum sequence length for RoPE (default: 2048)
        max_seq_len: Maximum sequence length for causal mask (default: 1024)
    """

    def __init__(
        self,
        embedding_dim,
        num_heads=8,
        ffn_hidden_dim=None,
        dropout=0.1,
        use_rope=False,
        rope_max_seq_len=2048,
        max_seq_len=1024
    ):
        super(GPT2DecoderBlock, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads

        # Layer normalization
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.ln2 = nn.LayerNorm(embedding_dim)

        # Causal self-attention
        self.attention = CausalQKVAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            max_seq_len=max_seq_len,
            use_rope=use_rope,
            rope_max_seq_len=rope_max_seq_len
        )

        # Feed-forward network
        self.ffn = FeedForward(
            embedding_dim=embedding_dim,
            hidden_dim=ffn_hidden_dim,
            dropout=dropout
        )

    def forward(self, x, mask=None):
        """
        Forward pass through the decoder block.

        Args:
            x: Input tensor of shape (batch_size, seq_len, embedding_dim)
            mask: Optional additional attention mask

        Returns:
            Output tensor of shape (batch_size, seq_len, embedding_dim)
        """
        # Self-attention with residual connection
        # x = x + attention(LayerNorm(x))
        attn_output = self.attention(self.ln1(x), mask=mask)
        x = x + attn_output

        # Feed-forward with residual connection
        # x = x + ffn(LayerNorm(x))
        ffn_output = self.ffn(self.ln2(x))
        x = x + ffn_output

        return x

    def save(self, filepath):
        """Save the decoder block weights."""
        torch.save(self.state_dict(), filepath)

    def load(self, filepath):
        """Load the decoder block weights."""
        self.load_state_dict(torch.load(filepath))


class GPT2Decoder(nn.Module):
    """
    Full GPT-2 decoder consisting of multiple transformer blocks.

    This stacks multiple GPT2DecoderBlock layers to create a deep transformer.
    Typically includes a final layer normalization after all blocks.

    Args:
        num_layers: Number of decoder blocks
        embedding_dim: Dimension of input embeddings
        num_heads: Number of attention heads per block
        ffn_hidden_dim: Hidden dimension for FFN (default: 4 * embedding_dim)
        dropout: Dropout probability (default: 0.1)
        use_rope: Whether to use Rotary Position Embeddings (default: False)
        rope_max_seq_len: Maximum sequence length for RoPE (default: 2048)
        max_seq_len: Maximum sequence length for causal mask (default: 1024)
    """

    def __init__(
        self,
        num_layers,
        embedding_dim,
        num_heads=8,
        ffn_hidden_dim=None,
        dropout=0.1,
        use_rope=False,
        rope_max_seq_len=2048,
        max_seq_len=1024
    ):
        super(GPT2Decoder, self).__init__()

        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads

        # Stack of decoder blocks
        self.blocks = nn.ModuleList([
            GPT2DecoderBlock(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                ffn_hidden_dim=ffn_hidden_dim,
                dropout=dropout,
                use_rope=use_rope,
                rope_max_seq_len=rope_max_seq_len,
                max_seq_len=max_seq_len
            )
            for _ in range(num_layers)
        ])

        # Final layer normalization
        self.ln_final = nn.LayerNorm(embedding_dim)

    def forward(self, x, mask=None):
        """
        Forward pass through all decoder blocks.

        Args:
            x: Input tensor of shape (batch_size, seq_len, embedding_dim)
            mask: Optional additional attention mask

        Returns:
            Output tensor of shape (batch_size, seq_len, embedding_dim)
        """
        # Pass through each decoder block
        for block in self.blocks:
            x = block(x, mask=mask)

        # Final layer normalization
        x = self.ln_final(x)

        return x

    def save(self, filepath):
        """Save the full decoder weights."""
        torch.save(self.state_dict(), filepath)

    def load(self, filepath):
        """Load the full decoder weights."""
        self.load_state_dict(torch.load(filepath))
