"""
MDLM (Masked Diffusion Language Model) in nanochat style.

Rewritten from nanochat/models/mdlm/modeling_mdlm.py following gpt.py patterns.

Notable features:
- Bidirectional attention (not causal) for diffusion modeling
- Timestep conditioning via AdaLN (Adaptive Layer Normalization)
- RoPE positional embeddings
- GELU activation in MLP
- Gated residual connections with dropout
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_dist_info, print0
from nanochat.optim import MuonAdamW, DistMuonAdamW
from nanochat.flash_attention import flash_attn


@dataclass
class MDLMConfig:
    sequence_len: int = 1024
    vocab_size: int = 50258
    n_layer: int = 12        # number of transformer blocks
    n_head: int = 12         # number of attention heads
    n_embd: int = 768        # embedding dimension (hidden_dim)
    cond_dim: int = 128      # timestep conditioning dimension
    mlp_ratio: int = 4       # MLP hidden dim = mlp_ratio * n_embd
    dropout: float = 0.1     # dropout probability
    time_conditioning: bool = True  # whether to condition on timesteps


def norm(x):
    """RMS normalization without learnable parameters."""
    return F.rms_norm(x, (x.size(-1),))


def modulate(x, shift, scale):
    """AdaLN modulation: x * (1 + scale) + shift."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def apply_rotary_emb(x, cos, sin):
    """Apply rotary positional embeddings to input tensor."""
    assert x.ndim == 4  # (B, T, H, D)
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], dim=3)


class LayerNorm(nn.Module):
    """LayerNorm with learnable weight (required for AdaLN modulation)."""
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.dim = dim

    def forward(self, x):
        with torch.amp.autocast('cuda', enabled=False):
            x = F.layer_norm(x.float(), (self.dim,))
        return x * self.weight


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations using sinusoidal encoding + MLP."""
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """Create sinusoidal timestep embeddings.

        Args:
            t: 1D tensor of N timestep indices (can be fractional)
            dim: output embedding dimension
            max_period: controls minimum frequency of embeddings

        Returns:
            (N, dim) tensor of positional embeddings
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class BidirectionalAttention(nn.Module):
    """Bidirectional self-attention with RoPE (for diffusion models)."""
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0

        # QKV projections (no bias, following GPT style)
        self.c_q = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

    def forward(self, x, cos_sin):
        B, T, C = x.size()

        # Project to Q, K, V
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_head, self.head_dim)

        # Apply RoPE to Q and K
        cos, sin = cos_sin
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # Bidirectional attention (causal=False)
        y = flash_attn.flash_attn_func(q, k, v, causal=False)

        # Project back
        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    """MLP with GELU activation."""
    def __init__(self, config):
        super().__init__()
        hidden_dim = config.mlp_ratio * config.n_embd
        self.c_fc = nn.Linear(config.n_embd, hidden_dim, bias=True)
        self.c_proj = nn.Linear(hidden_dim, config.n_embd, bias=True)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.gelu(x, approximate='tanh')
        x = self.c_proj(x)
        return x


class MDLMBlock(nn.Module):
    """Transformer block with AdaLN conditioning for diffusion."""
    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        # Layer norms (learnable, required for AdaLN)
        self.norm1 = LayerNorm(config.n_embd)
        self.norm2 = LayerNorm(config.n_embd)

        # Attention and MLP
        self.attn = BidirectionalAttention(config)
        self.mlp = MLP(config)

        # AdaLN modulation: 6 outputs (shift, scale, gate) for both attention and MLP
        self.adaLN_modulation = nn.Linear(config.cond_dim, 6 * config.n_embd, bias=True)

    def forward(self, x, cos_sin, c):
        """
        Args:
            x: input tensor (B, T, C)
            cos_sin: tuple of (cos, sin) for RoPE
            c: conditioning tensor (B, cond_dim) from timestep embedding
        """
        # Get modulation parameters from conditioning
        modulation = self.adaLN_modulation(c)  # (B, 6 * n_embd)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation.chunk(6, dim=-1)

        # Attention with AdaLN modulation
        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out = self.attn(x_norm, cos_sin)
        attn_out = F.dropout(attn_out, p=self.dropout, training=self.training)
        x = x + gate_msa.unsqueeze(1) * attn_out

        # MLP with AdaLN modulation
        x_norm = modulate(self.norm2(x), shift_mlp, scale_mlp)
        mlp_out = self.mlp(x_norm)
        mlp_out = F.dropout(mlp_out, p=self.dropout, training=self.training)
        x = x + gate_mlp.unsqueeze(1) * mlp_out

        return x


class MDLMFinalLayer(nn.Module):
    """Final layer with AdaLN modulation before output projection."""
    def __init__(self, config):
        super().__init__()
        self.norm = LayerNorm(config.n_embd)
        self.linear = nn.Linear(config.n_embd, config.vocab_size, bias=True)

        # AdaLN modulation: 2 outputs (shift, scale)
        self.adaLN_modulation = nn.Linear(config.cond_dim, 2 * config.n_embd, bias=True)

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm(x), shift, scale)
        x = self.linear(x)
        return x


class MDLM(nn.Module):
    """Masked Diffusion Language Model."""

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

        # Timestep embedding
        self.sigma_map = TimestepEmbedder(config.cond_dim)

        # Transformer blocks
        self.blocks = nn.ModuleList([MDLMBlock(config) for _ in range(config.n_layer)])

        # Output layer
        self.final_layer = MDLMFinalLayer(config)

        # Precompute rotary embeddings (10x over-allocation like GPT)
        self.rotary_seq_len = config.sequence_len * 10
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        """Initialize model weights following GPT patterns with MDLM-specific zero inits."""
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5  # uniform bound for same std as normal

        # Token embedding: normal with std=1.0
        torch.nn.init.normal_(self.wte.weight, mean=0.0, std=1.0)

        # Timestep embedder MLP
        for module in self.sigma_map.mlp:
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

        # Transformer blocks
        for block in self.blocks:
            # Attention projections: uniform init
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)

            # MLP: uniform init for fc, zeros for proj
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            if block.mlp.c_fc.bias is not None:
                torch.nn.init.zeros_(block.mlp.c_fc.bias)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
            if block.mlp.c_proj.bias is not None:
                torch.nn.init.zeros_(block.mlp.c_proj.bias)

            # AdaLN modulation: zero init (critical for training stability)
            torch.nn.init.zeros_(block.adaLN_modulation.weight)
            torch.nn.init.zeros_(block.adaLN_modulation.bias)

            # LayerNorm weights
            block.norm1.weight.fill_(1.0)
            block.norm2.weight.fill_(1.0)

        # Final layer
        torch.nn.init.zeros_(self.final_layer.linear.weight)
        torch.nn.init.zeros_(self.final_layer.linear.bias)
        torch.nn.init.zeros_(self.final_layer.adaLN_modulation.weight)
        torch.nn.init.zeros_(self.final_layer.adaLN_modulation.bias)
        self.final_layer.norm.weight.fill_(1.0)

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
        sigma_map = sum(p.numel() for p in self.sigma_map.parameters())
        blocks = sum(p.numel() for p in self.blocks.parameters())
        final_layer = sum(p.numel() for p in self.final_layer.parameters())
        total = wte + sigma_map + blocks + final_layer

        assert total == sum(p.numel() for p in self.parameters()), "Parameter count mismatch"
        return {
            'wte': wte,
            'sigma_map': sigma_map,
            'blocks': blocks,
            'final_layer': final_layer,
            'total': total,
        }

    def setup_optimizer(self, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, adam_betas=(0.8, 0.95)):
        """Setup MuonAdamW optimizer with parameter groups."""
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()

        # Collect parameters by type
        embedding_params = list(self.wte.parameters())
        sigma_map_params = list(self.sigma_map.parameters())

        # Separate matrix params from blocks and final layer
        matrix_params = []
        adaLN_params = []  # AdaLN modulation layers (small, use AdamW)
        norm_params = []   # LayerNorm weights

        for block in self.blocks:
            matrix_params.extend([
                block.attn.c_q.weight, block.attn.c_k.weight,
                block.attn.c_v.weight, block.attn.c_proj.weight,
                block.mlp.c_fc.weight, block.mlp.c_proj.weight,
            ])
            if block.mlp.c_fc.bias is not None:
                adaLN_params.append(block.mlp.c_fc.bias)
            if block.mlp.c_proj.bias is not None:
                adaLN_params.append(block.mlp.c_proj.bias)
            adaLN_params.extend([block.adaLN_modulation.weight, block.adaLN_modulation.bias])
            norm_params.extend([block.norm1.weight, block.norm2.weight])

        # Final layer
        adaLN_params.extend([
            self.final_layer.linear.weight, self.final_layer.linear.bias,
            self.final_layer.adaLN_modulation.weight, self.final_layer.adaLN_modulation.bias,
        ])
        norm_params.append(self.final_layer.norm.weight)

        # Scale LR for AdamW by 1/sqrt(dmodel)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling AdamW LR by 1/sqrt({model_dim}/768) = {dmodel_lr_scale:.6f}")

        # Build param groups
        param_groups = [
            # AdamW groups
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=sigma_map_params, lr=embedding_lr * dmodel_lr_scale * 0.1, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=adaLN_params, lr=embedding_lr * dmodel_lr_scale * 0.1, betas=adam_betas, eps=1e-10, weight_decay=0.0),
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

    def forward(self, input_ids, timesteps=None):
        """
        Forward pass for MDLM.

        Args:
            input_ids: token indices (B, T)
            timesteps: diffusion timesteps (B,) - scalar per batch element

        Returns:
            logits: (B, T, vocab_size)
        """
        B, T = input_ids.size()

        # Handle time conditioning
        if not self.config.time_conditioning or timesteps is None:
            timesteps = torch.zeros(B, device=input_ids.device)

        # Get rotary embeddings for current sequence
        assert T <= self.cos.size(1), f"Sequence too long: {T} > {self.cos.size(1)}"
        cos_sin = self.cos[:, :T], self.sin[:, :T]

        # Embed tokens
        x = self.wte(input_ids)

        # Embed timesteps and apply SiLU
        c = F.silu(self.sigma_map(timesteps))

        # Forward through transformer blocks
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            for block in self.blocks:
                x = block(x, cos_sin, c)

            # Final layer
            logits = self.final_layer(x, c)

        # Crop to actual vocab size (remove padding)
        logits = logits[..., :self.config.vocab_size]

        return logits
