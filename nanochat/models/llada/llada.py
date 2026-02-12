"""
LLaDA (Large Language Diffusion with mAsking) in nanochat style.

Rewritten from nanochat/models/llada/modeling_llada.py following gpt.py patterns.

Notable features:
- Bidirectional attention (not causal) for masked diffusion
- RoPE positional embeddings
- SwiGLU activation (LLaMA-style gated MLP)
- Optional QK normalization
- GQA (Group Query Attention) support
- Dropout at embedding, attention, and residual connections
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_dist_info, print0
from nanochat.optim import MuonAdamW, DistMuonAdamW
from nanochat.flash_attention import flash_attn


@dataclass
class LLaDAConfig:
    sequence_len: int = 1024
    vocab_size: int = 50258
    n_layer: int = 12        # number of transformer blocks
    n_head: int = 12         # number of query heads
    n_kv_head: int = 12      # number of key/value heads (for GQA)
    n_embd: int = 768        # embedding dimension
    mlp_ratio: int = 4       # MLP hidden dim multiplier (actual = 8/3 * n_embd for SwiGLU)
    dropout: float = 0.1     # dropout probability
    qk_norm: bool = True     # apply layer norm to Q and K


def norm(x):
    """RMS normalization without learnable parameters."""
    return F.rms_norm(x, (x.size(-1),))


def apply_rotary_emb(x, cos, sin):
    """Apply rotary positional embeddings to input tensor."""
    assert x.ndim == 4  # (B, T, H, D)
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], dim=3)


class RMSNorm(nn.Module):
    """RMS LayerNorm with learnable weight."""
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        with torch.amp.autocast('cuda', enabled=False):
            x_float = x.float()
            variance = x_float.pow(2).mean(-1, keepdim=True)
            x_normed = x_float * torch.rsqrt(variance + self.eps)
        return self.weight * x_normed.to(x.dtype)


class BidirectionalAttention(nn.Module):
    """Bidirectional self-attention with RoPE and GQA support."""
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.qk_norm = config.qk_norm
        self.dropout = config.dropout

        assert self.n_embd % self.n_head == 0
        assert self.n_head % self.n_kv_head == 0

        # Separate Q/K/V projections (LLaMA-style)
        self.q_proj = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

        # Optional QK normalization
        if self.qk_norm:
            self.q_norm = RMSNorm(self.n_embd)
            self.k_norm = RMSNorm(self.n_kv_head * self.head_dim)

    def forward(self, x, cos_sin):
        B, T, C = x.size()

        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Optional QK normalization
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # Reshape for attention
        q = q.view(B, T, self.n_head, self.head_dim)
        k = k.view(B, T, self.n_kv_head, self.head_dim)
        v = v.view(B, T, self.n_kv_head, self.head_dim)

        # Apply RoPE to Q and K
        cos, sin = cos_sin
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # Bidirectional attention (causal=False)
        y = flash_attn.flash_attn_func(q, k, v, causal=False)

        # Project back
        y = y.contiguous().view(B, T, -1)
        y = self.o_proj(y)
        return y


class SwiGLUMLP(nn.Module):
    """LLaMA-style gated MLP with SwiGLU activation."""
    def __init__(self, config):
        super().__init__()
        # For SwiGLU, hidden_dim is typically 8/3 * n_embd to match param count
        # But we use mlp_ratio for flexibility
        hidden_dim = int((8 / 3) * config.n_embd)
        # Round to nearest multiple of 64 for efficiency
        hidden_dim = ((hidden_dim + 63) // 64) * 64

        self.gate_proj = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.up_proj = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, config.n_embd, bias=False)

    def forward(self, x):
        # SwiGLU: silu(gate) * up
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class LLaDABlock(nn.Module):
    """Transformer block for LLaDA."""
    def __init__(self, config):
        super().__init__()
        self.dropout_p = config.dropout

        # Layer norms
        self.attn_norm = RMSNorm(config.n_embd)
        self.ffn_norm = RMSNorm(config.n_embd)

        # Attention and MLP
        self.attn = BidirectionalAttention(config)
        self.mlp = SwiGLUMLP(config)

    def forward(self, x, cos_sin):
        # Attention with residual
        h = self.attn_norm(x)
        h = self.attn(h, cos_sin)
        h = F.dropout(h, p=self.dropout_p, training=self.training)
        x = x + h

        # MLP with residual
        h = self.ffn_norm(x)
        h = self.mlp(h)
        h = F.dropout(h, p=self.dropout_p, training=self.training)
        x = x + h

        return x


class LLaDA(nn.Module):
    """Large Language Diffusion with mAsking model."""

    def __init__(self, config, pad_vocab_size_to=64):
        super().__init__()
        self.config = config

        # Pad vocab for efficiency (DDP, tensor cores)
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")
        self.padded_vocab_size = padded_vocab_size

        # Token embedding
        self.wte = nn.Embedding(padded_vocab_size, config.n_embd)

        # Embedding dropout
        self.emb_dropout = config.dropout

        # Transformer blocks
        self.blocks = nn.ModuleList([LLaDABlock(config) for _ in range(config.n_layer)])

        # Final layer norm
        self.ln_f = RMSNorm(config.n_embd)

        # Output projection (untied from input embedding)
        self.lm_head = nn.Linear(config.n_embd, padded_vocab_size, bias=False)

        # Precompute rotary embeddings (10x over-allocation like GPT)
        self.rotary_seq_len = config.sequence_len * 10
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        """Initialize model weights following GPT patterns."""
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5  # uniform bound for same std as normal

        # Token embedding: normal with std=1.0
        torch.nn.init.normal_(self.wte.weight, mean=0.0, std=1.0)

        # Output projection: small init
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        # Transformer blocks
        for block in self.blocks:
            # Attention projections: uniform init
            torch.nn.init.uniform_(block.attn.q_proj.weight, -s, s)
            torch.nn.init.uniform_(block.attn.k_proj.weight, -s, s)
            torch.nn.init.uniform_(block.attn.v_proj.weight, -s, s)
            torch.nn.init.zeros_(block.attn.o_proj.weight)

            # QK norm weights
            if block.attn.qk_norm:
                block.attn.q_norm.weight.fill_(1.0)
                block.attn.k_norm.weight.fill_(1.0)

            # MLP projections: uniform init, down_proj zero
            torch.nn.init.uniform_(block.mlp.gate_proj.weight, -s, s)
            torch.nn.init.uniform_(block.mlp.up_proj.weight, -s, s)
            torch.nn.init.zeros_(block.mlp.down_proj.weight)

            # Layer norm weights
            block.attn_norm.weight.fill_(1.0)
            block.ffn_norm.weight.fill_(1.0)

        # Final layer norm
        self.ln_f.weight.fill_(1.0)

        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

        # Cast embeddings to bf16 if on CUDA
        if self.wte.weight.device.type == "cuda":
            self.wte.to(dtype=torch.bfloat16)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        """Precompute rotary embeddings for given sequence length."""
        if device is None:
            device = self.wte.weight.device

        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))

        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16()
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]  # (1, T, 1, D/2)
        return cos, sin

    def get_device(self):
        return self.wte.weight.device

    def estimate_flops(self):
        """Return estimated FLOPs per token for forward + backward pass."""
        # Count matmul parameters (excluding embeddings)
        nparams = sum(p.numel() for p in self.parameters())
        nparams_exclude = self.wte.weight.numel()

        # Attention FLOPs: bidirectional, full sequence
        h = self.config.n_head
        q = self.config.n_embd // self.config.n_head
        t = self.config.sequence_len
        attn_flops = self.config.n_layer * 12 * h * q * t  # full context for all layers

        num_flops_per_token = 6 * (nparams - nparams_exclude) + attn_flops
        return num_flops_per_token

    def num_scaling_params(self):
        """Return detailed parameter counts for scaling law analysis."""
        wte = self.wte.weight.numel()
        lm_head = self.lm_head.weight.numel()
        blocks = sum(p.numel() for p in self.blocks.parameters())
        ln_f = self.ln_f.weight.numel()
        total = wte + lm_head + blocks + ln_f

        assert total == sum(p.numel() for p in self.parameters()), "Parameter count mismatch"
        return {
            'wte': wte,
            'lm_head': lm_head,
            'blocks': blocks,
            'ln_f': ln_f,
            'total': total,
        }

    def setup_optimizer(self, embedding_lr=0.2, unembedding_lr=0.004, matrix_lr=0.02, weight_decay=0.0, adam_betas=(0.8, 0.95)):
        """Setup MuonAdamW optimizer with parameter groups."""
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()

        # Collect parameters by type
        embedding_params = list(self.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())

        # Separate matrix params from blocks
        matrix_params = []
        norm_params = []

        for block in self.blocks:
            matrix_params.extend([
                block.attn.q_proj.weight, block.attn.k_proj.weight,
                block.attn.v_proj.weight, block.attn.o_proj.weight,
                block.mlp.gate_proj.weight, block.mlp.up_proj.weight,
                block.mlp.down_proj.weight,
            ])
            norm_params.extend([block.attn_norm.weight, block.ffn_norm.weight])
            if block.attn.qk_norm:
                norm_params.extend([block.attn.q_norm.weight, block.attn.k_norm.weight])

        norm_params.append(self.ln_f.weight)

        # Scale LR for AdamW by 1/sqrt(dmodel)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling AdamW LR by 1/sqrt({model_dim}/768) = {dmodel_lr_scale:.6f}")

        # Build param groups
        param_groups = [
            # AdamW groups
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=norm_params, lr=embedding_lr * dmodel_lr_scale * 0.1, betas=adam_betas, eps=1e-10, weight_decay=0.0),
        ]

        # Muon groups (matrix params, grouped by shape)
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,
            ))

        Factory = DistMuonAdamW if ddp else MuonAdamW
        optimizer = Factory(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass for LLaDA.

        Args:
            input_ids: token indices (B, T)
            attention_mask: optional mask (B, T) - not used for bidirectional attention

        Returns:
            logits: (B, T, vocab_size)
        """
        B, T = input_ids.size()

        # Get rotary embeddings for current sequence
        assert T <= self.cos.size(1), f"Sequence too long: {T} > {self.cos.size(1)}"
        cos_sin = self.cos[:, :T], self.sin[:, :T]

        # Embed tokens
        x = self.wte(input_ids)

        # Embedding dropout
        x = F.dropout(x, p=self.emb_dropout, training=self.training)

        # Forward through transformer blocks
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            for block in self.blocks:
                x = block(x, cos_sin)

            # Final layer norm
            x = self.ln_f(x)

            # Output projection
            logits = self.lm_head(x)

        # Crop to actual vocab size (remove padding)
        logits = logits[..., :self.config.vocab_size]

        return logits
