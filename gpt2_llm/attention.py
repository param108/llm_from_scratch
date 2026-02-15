"""
Query, Key, Value attention mechanisms for transformer models.
"""

import torch
import torch.nn as nn
import math


class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) from RoFormer paper.

    RoPE encodes positional information by rotating Q and K vectors.
    This allows the model to capture relative positions naturally through
    the dot product operation.

    Args:
        dim: Dimension of the embeddings (should be head_dim)
        max_seq_len: Maximum sequence length (default: 2048)
        base: Base for the frequency computation (default: 10000)
    """

    def __init__(self, dim, max_seq_len=2048, base=10000):
        super(RotaryPositionEmbedding, self).__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Compute frequency bands
        # inv_freq shape: (dim // 2,)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        # Pre-compute cos and sin for efficiency
        self._seq_len_cached = None
        self._cos_cached = None
        self._sin_cached = None

    def _update_cos_sin_cache(self, seq_len, device):
        """Update the cached cos and sin values if sequence length changes."""
        if seq_len != self._seq_len_cached:
            self._seq_len_cached = seq_len
            # Create position indices: [0, 1, 2, ..., seq_len-1]
            t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
            # Compute angles: outer product of positions and frequencies
            # Shape: (seq_len, dim // 2)
            freqs = torch.outer(t, self.inv_freq)
            # Create rotation matrix values
            # Shape: (seq_len, dim)
            emb = torch.cat([freqs, freqs], dim=-1)
            self._cos_cached = emb.cos()
            self._sin_cached = emb.sin()

    def _rotate_half(self, x):
        """
        Rotate half the hidden dims of the input.

        For a vector [x1, x2, x3, x4], this returns [-x3, -x4, x1, x2]
        This is used to apply the rotation in the RoPE formula.
        """
        x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, q, k):
        """
        Apply rotary position embedding to Q and K.

        Args:
            q: Query tensor of shape (batch, num_heads, seq_len, head_dim)
            k: Key tensor of shape (batch, num_heads, seq_len, head_dim)

        Returns:
            Tuple of (q_rotated, k_rotated) with same shapes as inputs
        """
        seq_len = q.shape[2]
        self._update_cos_sin_cache(seq_len, q.device)

        # Get cos and sin for current sequence
        # Shape: (seq_len, dim) -> (1, 1, seq_len, dim)
        cos = self._cos_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        sin = self._sin_cached[:seq_len].unsqueeze(0).unsqueeze(0)

        # Apply rotation: x * cos + rotate_half(x) * sin
        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)

        return q_embed, k_embed


class QKVAttention(nn.Module):
    """
    Multi-head attention with separate Query, Key, Value projections.

    This implements the attention mechanism from "Attention is All You Need":
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V

    Args:
        embedding_dim: Dimension of input embeddings
        num_heads: Number of attention heads
        dropout: Dropout probability (default: 0.1)
        use_rope: Whether to use Rotary Position Embeddings (default: False)
        rope_max_seq_len: Maximum sequence length for RoPE (default: 2048)
    """

    def __init__(self, embedding_dim, num_heads=8, dropout=0.1, use_rope=False, rope_max_seq_len=2048):
        super(QKVAttention, self).__init__()

        if embedding_dim % num_heads != 0:
            raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by num_heads ({num_heads})")

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        self.use_rope = use_rope

        # Linear projections for Q, K, V
        self.query_proj = nn.Linear(embedding_dim, embedding_dim)
        self.key_proj = nn.Linear(embedding_dim, embedding_dim)
        self.value_proj = nn.Linear(embedding_dim, embedding_dim)

        # Output projection
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Rotary Position Embedding (optional)
        if use_rope:
            self.rope = RotaryPositionEmbedding(self.head_dim, max_seq_len=rope_max_seq_len)
        else:
            self.rope = None

    def forward(self, x, mask=None):
        """
        Forward pass for multi-head attention.

        Args:
            x: Input tensor of shape (batch_size, seq_len, embedding_dim)
            mask: Optional attention mask of shape (batch_size, seq_len, seq_len)
                  or (seq_len, seq_len). Use -inf for positions to mask out.

        Returns:
            Output tensor of shape (batch_size, seq_len, embedding_dim)
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        Q = self.query_proj(x)  # (batch_size, seq_len, embedding_dim)
        K = self.key_proj(x)
        V = self.value_proj(x)

        # Reshape for multi-head attention
        # (batch_size, seq_len, num_heads, head_dim) -> (batch_size, num_heads, seq_len, head_dim)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply Rotary Position Embedding if enabled
        if self.use_rope:
            Q, K = self.rope(Q, K)

        # Compute attention scores: QK^T / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        # Shape: (batch_size, num_heads, seq_len, seq_len)

        # Apply mask if provided
        if mask is not None:
            # Expand mask to match attention scores shape
            if mask.dim() == 2:  # (seq_len, seq_len)
                mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
            elif mask.dim() == 3:  # (batch_size, seq_len, seq_len)
                mask = mask.unsqueeze(1)  # (batch_size, 1, seq_len, seq_len)
            scores = scores + mask

        # Apply softmax
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)
        # Shape: (batch_size, num_heads, seq_len, head_dim)

        # Reshape back to (batch_size, seq_len, embedding_dim)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.embedding_dim)

        # Final output projection
        output = self.out_proj(attn_output)

        return output

    def save(self, filepath):
        """Save the attention layer weights."""
        torch.save(self.state_dict(), filepath)

    def load(self, filepath):
        """Load the attention layer weights."""
        self.load_state_dict(torch.load(filepath))


class CausalQKVAttention(QKVAttention):
    """
    Causal (autoregressive) multi-head attention for GPT-2 style models.

    This prevents positions from attending to future positions by applying
    a causal mask during attention computation.

    Args:
        embedding_dim: Dimension of input embeddings
        num_heads: Number of attention heads
        dropout: Dropout probability (default: 0.1)
        max_seq_len: Maximum sequence length for pre-computed causal mask (default: 1024)
        use_rope: Whether to use Rotary Position Embeddings (default: False)
        rope_max_seq_len: Maximum sequence length for RoPE (default: 2048)
    """

    def __init__(self, embedding_dim, num_heads=8, dropout=0.1, max_seq_len=1024, use_rope=False, rope_max_seq_len=2048):
        super(CausalQKVAttention, self).__init__(embedding_dim, num_heads, dropout, use_rope, rope_max_seq_len)

        # Register causal mask as a buffer (not a parameter, but part of state)
        # Lower triangular matrix: positions can attend to themselves and earlier positions
        causal_mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        # Convert to attention mask: 0 -> -inf, 1 -> 0
        causal_mask = causal_mask.masked_fill(causal_mask == 0, float('-inf'))
        causal_mask = causal_mask.masked_fill(causal_mask == 1, 0.0)
        self.register_buffer('causal_mask', causal_mask)

    def forward(self, x, mask=None):
        """
        Forward pass with causal masking.

        Args:
            x: Input tensor of shape (batch_size, seq_len, embedding_dim)
            mask: Optional additional mask to combine with causal mask

        Returns:
            Output tensor of shape (batch_size, seq_len, embedding_dim)
        """
        seq_len = x.shape[1]

        # Get causal mask for current sequence length
        causal_mask = self.causal_mask[:seq_len, :seq_len]

        # Combine with additional mask if provided
        if mask is not None:
            causal_mask = causal_mask + mask

        # Call parent forward with causal mask
        return super().forward(x, causal_mask)
