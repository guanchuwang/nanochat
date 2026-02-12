"""
Energy-Based Transformer (EBT) for NLP
Rewritten in nanochat style following gpt.py conventions.

Notable features:
- MCMC-based next token prediction via energy minimization
- Learnable step size (alpha) and optional Langevin dynamics noise
- Causal attention with rotary embeddings
- Vocab-to-embedding projection for probability distributions
- Energy output (scalar per position) instead of logits

Reference: https://arxiv.org/abs/2406.08862
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_dist_info, print0
from nanochat.optim import MuonAdamW, DistMuonAdamW
from nanochat.flash_attention import flash_attn


@dataclass
class EBTConfig:
    # Model architecture
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 12
    n_kv_head: int = 12
    n_embd: int = 768

    # MCMC hyperparameters
    mcmc_num_steps: int = 10
    mcmc_step_size: float = 0.01
    mcmc_step_size_learnable: bool = True
    langevin_dynamics_noise: float = 0.0
    langevin_dynamics_noise_learnable: bool = False

    # MCMC behavior
    normalize_initial_condition: bool = True
    normalize_initial_condition_only_first_step: bool = False
    vocab_to_embed_uses_prob_dist: bool = True
    denoising_initial_condition: str = "random_noise"  # "random_noise" or "zeros"
    gaussian_random_noise_scaling: float = 1.0

    # Gradient control
    clamp_futures_grad: bool = True
    clamp_futures_grad_max_change: float = 1.0
    absolute_clamp: float = 0.0
    truncate_mcmc: bool = True
    no_mcmc_detach: bool = False

    # Distribution sharpening (temperature)
    sharpen_predicted_distribution: float = 0.0

    # Loss
    reconstruction_coeff: float = 1.0
    soften_target_prob_dist: float = 0.0


def norm(x):
    """Purely functional RMS norm with no learnable params."""
    return F.rms_norm(x, (x.size(-1),))


def apply_rotary_emb(x, cos, sin):
    """Apply rotary position embeddings to input tensor."""
    assert x.ndim == 4  # (B, T, H, D)
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with rotary embeddings and QK norm."""

    def __init__(self, config: EBTConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head

        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0

        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

    def forward(self, x, cos_sin):
        B, T, C = x.size()

        # Project to Q, K, V
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Apply rotary embeddings
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)  # QK norm

        # Flash Attention (causal)
        y = flash_attn.flash_attn_func(q, k, v, causal=True)

        # Project back
        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    """Feed-forward network with ReLU^2 activation."""

    def __init__(self, config: EBTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()  # ReLU^2 activation
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    """Transformer block: attention + MLP with pre-norm."""

    def __init__(self, config: EBTConfig, layer_idx: int):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, cos_sin):
        x = x + self.attn(norm(x), cos_sin)
        x = x + self.mlp(norm(x))
        return x


class EnergyHead(nn.Module):
    """
    Projects transformer output to scalar energy values.
    Outputs one energy value per sequence position.
    """

    def __init__(self, config: EBTConfig):
        super().__init__()
        self.proj = nn.Linear(config.n_embd, 1, bias=False)

    def forward(self, x):
        # x: (B, T, D) -> (B, T, 1) -> sum over T -> (B, 1)
        energy = self.proj(x)  # (B, T, 1)
        energy = energy.sum(dim=1)  # (B, 1) - sum energy across sequence
        return energy


class EBT(nn.Module):
    """
    Energy-Based Transformer for next token prediction via MCMC.

    Instead of directly predicting logits, EBT learns an energy function over
    (context, next_token_distribution) pairs. Lower energy = more likely.
    Training uses MCMC to find low-energy predictions, then supervises with
    cross-entropy loss.
    """

    def __init__(self, config: EBTConfig, pad_vocab_size_to: int = 64):
        super().__init__()
        self.config = config

        # Learnable MCMC parameters
        self.alpha = nn.Parameter(
            torch.tensor(float(config.mcmc_step_size)),
            requires_grad=config.mcmc_step_size_learnable
        )
        self.langevin_dynamics_noise_std = nn.Parameter(
            torch.tensor(float(config.langevin_dynamics_noise)),
            requires_grad=False  # turned on later via warm_up_finished() if needed
        )

        # Pad vocab for efficiency
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")
        self.padded_vocab_size = padded_vocab_size

        # Token embeddings
        self.embeddings = nn.Embedding(padded_vocab_size, config.n_embd)

        # Vocab-to-embedding projection (for probability distributions)
        # Maps (B, S, V) -> (B, S, D)
        if not config.vocab_to_embed_uses_prob_dist:
            self.vocab_to_embed = nn.Linear(padded_vocab_size, config.n_embd, bias=False)
        else:
            self.vocab_to_embed = None  # Use matmul with embedding weights instead

        # Transformer backbone
        self.transformer = nn.ModuleDict({
            "h": nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
        })

        # Energy head: projects to scalar energy
        self.energy_head = EnergyHead(config)

        # Rotary embeddings (pre-computed, 10x over-allocation)
        self.rotary_seq_len = config.sequence_len * 10
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        # Softmax/LogSoftmax for distribution operations
        self.softmax = nn.Softmax(dim=-1)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def _precompute_rotary_embeddings(self, seq_len: int, head_dim: int, base: int = 10000, device=None):
        """Precompute rotary position embeddings."""
        if device is None:
            device = self.embeddings.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16()
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin

    @torch.no_grad()
    def init_weights(self):
        """Initialize all weights following gpt.py conventions."""
        # Token embeddings
        torch.nn.init.normal_(self.embeddings.weight, mean=0.0, std=1.0)

        # Vocab-to-embed projection (if used)
        if self.vocab_to_embed is not None:
            torch.nn.init.xavier_normal_(self.vocab_to_embed.weight)

        # Energy head
        torch.nn.init.normal_(self.energy_head.proj.weight, mean=0.0, std=0.001)

        # Transformer blocks
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)

        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

        # Cast embeddings to bf16
        if self.embeddings.weight.device.type == "cuda":
            self.embeddings.to(dtype=torch.bfloat16)

    def get_device(self):
        return self.embeddings.weight.device

    def corrupt_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Initialize predicted token logits for MCMC.

        Args:
            embeddings: Real token embeddings (B, S, D)

        Returns:
            Initial predicted logits (B, S, V)
        """
        B, S, D = embeddings.shape
        V = self.config.vocab_size
        device = embeddings.device

        if self.config.denoising_initial_condition == "random_noise":
            predicted_tokens = torch.randn(B, S, V, device=device) * self.config.gaussian_random_noise_scaling
        elif self.config.denoising_initial_condition == "zeros":
            predicted_tokens = torch.zeros(B, S, V, device=device)
        else:
            raise ValueError(f"Unknown denoising_initial_condition: {self.config.denoising_initial_condition}")

        return predicted_tokens

    def logits_to_embeddings(self, predicted_tokens: torch.Tensor, mcmc_step: int) -> torch.Tensor:
        """
        Convert predicted logits to embeddings.

        Args:
            predicted_tokens: Predicted token logits (B, S, V)
            mcmc_step: Current MCMC step (for conditional normalization)

        Returns:
            Predicted embeddings (B, S, D)
        """
        # Optionally normalize to probability distribution
        if self.config.normalize_initial_condition:
            if self.config.normalize_initial_condition_only_first_step:
                if mcmc_step == 0:
                    predicted_tokens = self.softmax(predicted_tokens)
            else:
                predicted_tokens = self.softmax(predicted_tokens)

            # Convert to embeddings
            if self.config.vocab_to_embed_uses_prob_dist:
                # Weighted sum of embeddings: (B, S, V) @ (V, D) -> (B, S, D)
                predicted_embeddings = torch.matmul(predicted_tokens, self.embeddings.weight[:self.config.vocab_size])
            else:
                predicted_embeddings = self.vocab_to_embed(predicted_tokens)
        else:
            predicted_embeddings = self.vocab_to_embed(predicted_tokens)

        return predicted_embeddings

    def compute_energy(self, all_embeddings: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        """
        Compute energy for concatenated (real, predicted) embeddings.

        Args:
            all_embeddings: Concatenated embeddings (B, 2*S, D)
            start_pos: Position offset for rotary embeddings

        Returns:
            Energy values (B, 1)
        """
        B, T, C = all_embeddings.size()

        # Get rotary embeddings for current sequence
        assert T <= self.cos.size(1), f"Sequence too long: {T} > {self.cos.size(1)}"
        cos_sin = self.cos[:, start_pos:start_pos+T], self.sin[:, start_pos:start_pos+T]

        # Forward through transformer
        x = norm(all_embeddings)
        for block in self.transformer.h:
            x = block(x, cos_sin)
        x = norm(x)

        # Compute energy
        energy = self.energy_head(x)  # (B, 1)
        return energy

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int = 0,
        learning: bool = True,
        no_randomness: bool = True,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass with MCMC dynamics.

        Args:
            x: Input token IDs (B, S)
            start_pos: Position offset for rotary embeddings
            learning: Whether gradients should flow through MCMC
            no_randomness: If True, disable random perturbations

        Returns:
            predicted_distributions: List of predicted log-prob distributions per MCMC step
            predicted_energies: List of energy values per MCMC step
        """
        predicted_distributions = []
        predicted_energies = []

        # Get real embeddings from input tokens
        real_embeddings_input = self.embeddings(x)  # (B, S, D)
        B, S, D = real_embeddings_input.shape
        V = self.config.vocab_size

        # Clamp alpha to positive
        alpha = torch.clamp(self.alpha, min=0.0001)
        langevin_noise_std = torch.clamp(self.langevin_dynamics_noise_std, min=0.000001)

        # Initialize predicted token logits
        predicted_tokens = self.corrupt_embeddings(real_embeddings_input)  # (B, S, V)

        # MCMC iterations
        with torch.set_grad_enabled(True):
            for mcmc_step in range(self.config.mcmc_num_steps):
                # Detach for gradient computation (unless no_mcmc_detach)
                if self.config.no_mcmc_detach:
                    predicted_tokens = predicted_tokens.requires_grad_().reshape(B, S, V)
                else:
                    predicted_tokens = predicted_tokens.detach().requires_grad_().reshape(B, S, V)

                # Add Langevin dynamics noise
                if self.config.langevin_dynamics_noise != 0 and not no_randomness:
                    noise = torch.randn_like(predicted_tokens) * langevin_noise_std
                    predicted_tokens = predicted_tokens + noise

                # Convert logits to embeddings
                predicted_embeddings = self.logits_to_embeddings(predicted_tokens, mcmc_step)

                # Concatenate real and predicted embeddings
                all_embeddings = torch.cat([real_embeddings_input, predicted_embeddings], dim=1)  # (B, 2*S, D)

                # Compute energy
                energy = self.compute_energy(all_embeddings, start_pos=start_pos)  # (B, 1)
                predicted_energies.append(energy)

                # Compute gradient of energy w.r.t. predicted_tokens
                is_last_step = (mcmc_step == self.config.mcmc_num_steps - 1)
                if self.config.truncate_mcmc:
                    # Only create graph on last step (for efficiency)
                    create_graph = learning and is_last_step
                else:
                    create_graph = learning

                grad = torch.autograd.grad(
                    energy.sum(), predicted_tokens, create_graph=create_graph
                )[0]

                # Gradient clipping
                if self.config.clamp_futures_grad:
                    clamp_val = self.config.clamp_futures_grad_max_change / self.alpha.item()
                    grad = torch.clamp(grad, min=-clamp_val, max=clamp_val)

                # Check for NaN/Inf
                if torch.isnan(grad).any() or torch.isinf(grad).any():
                    raise ValueError("NaN or Inf gradients detected during MCMC")

                # Gradient descent update
                predicted_tokens = predicted_tokens - alpha * grad

                # Absolute clamp
                if self.config.absolute_clamp != 0.0:
                    predicted_tokens = torch.clamp(
                        predicted_tokens,
                        min=-self.config.absolute_clamp,
                        max=self.config.absolute_clamp
                    )

                # Temperature sharpening
                if self.config.sharpen_predicted_distribution != 0.0:
                    predicted_tokens = predicted_tokens / self.config.sharpen_predicted_distribution

                # Store predicted distribution
                pred_log_probs = self.log_softmax(predicted_tokens).reshape(-1, V)  # (B*S, V)
                predicted_distributions.append(pred_log_probs)

        return predicted_distributions, predicted_energies

    def forward_loss(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor,
        phase: str = "train",
        ignore_index: int = -1,
    ) -> dict:
        """
        Compute loss for training.

        Args:
            input_ids: Input token IDs (B, S) - context
            targets: Target token IDs (B, S) - next tokens
            phase: "train" or "val"
            ignore_index: Index to ignore in loss computation

        Returns:
            Dictionary with loss values and metrics
        """
        no_randomness = (phase != "train")

        # Forward pass
        predicted_distributions, predicted_energies = self(
            input_ids, learning=True, no_randomness=no_randomness
        )

        # Flatten targets
        targets_flat = targets.reshape(-1)

        # Compute loss over MCMC steps
        reconstruction_loss = 0.0
        total_mcmc_steps = len(predicted_energies)

        for mcmc_step, (pred_dist, pred_energy) in enumerate(zip(predicted_distributions, predicted_energies)):
            # Label smoothing based on MCMC step (optional)
            if self.config.soften_target_prob_dist != 0.0 and total_mcmc_steps > 1:
                # More smoothing at earlier steps
                label_smoothing = ((total_mcmc_steps - 1) - mcmc_step) / (total_mcmc_steps - 1) * self.config.soften_target_prob_dist
                loss = F.cross_entropy(
                    pred_dist,
                    targets_flat,
                    ignore_index=ignore_index,
                    label_smoothing=label_smoothing
                )
            else:
                loss = F.nll_loss(pred_dist, targets_flat, ignore_index=ignore_index)

            if self.config.truncate_mcmc:
                # Only use final step loss
                if mcmc_step == total_mcmc_steps - 1:
                    reconstruction_loss = loss
            else:
                reconstruction_loss += loss

        # Normalize if not truncating
        if not self.config.truncate_mcmc:
            reconstruction_loss = reconstruction_loss / total_mcmc_steps

        # Final metrics
        final_loss = F.nll_loss(predicted_distributions[-1], targets_flat, ignore_index=ignore_index)
        perplexity = torch.exp(final_loss).detach()

        total_loss = self.config.reconstruction_coeff * reconstruction_loss

        return {
            'loss': total_loss,
            'final_step_loss': final_loss.detach(),
            'perplexity': perplexity,
            'initial_energy': predicted_energies[0].mean().detach() if predicted_energies else 0.0,
            'final_energy': predicted_energies[-1].mean().detach() if predicted_energies else 0.0,
        }

    def setup_optimizer(
        self,
        embedding_lr: float = 0.2,
        matrix_lr: float = 0.02,
        mcmc_lr: float = 0.001,
        weight_decay: float = 0.0,
        adam_betas: Tuple[float, float] = (0.8, 0.95),
    ):
        """
        Set up optimizer following gpt.py conventions.
        Uses Muon for transformer matrices, AdamW for embeddings and MCMC params.
        """
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()

        # Separate parameters
        matrix_params = list(self.transformer.h.parameters())
        embedding_params = list(self.embeddings.parameters())
        energy_head_params = list(self.energy_head.parameters())

        # MCMC parameters (alpha, langevin noise)
        mcmc_params = [self.alpha]
        if self.langevin_dynamics_noise_std.requires_grad:
            mcmc_params.append(self.langevin_dynamics_noise_std)

        # Vocab-to-embed if exists
        v2e_params = list(self.vocab_to_embed.parameters()) if self.vocab_to_embed is not None else []

        # Scale LR by model dimension
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling AdamW LRs by 1/sqrt({model_dim}/768) = {dmodel_lr_scale:.6f}")

        param_groups = [
            # AdamW groups
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale,
                 betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=energy_head_params, lr=0.004 * dmodel_lr_scale,
                 betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=mcmc_params, lr=mcmc_lr,
                 betas=adam_betas, eps=1e-10, weight_decay=0.0),
        ]

        if v2e_params:
            param_groups.append(
                dict(kind='adamw', params=v2e_params, lr=embedding_lr * dmodel_lr_scale,
                     betas=adam_betas, eps=1e-10, weight_decay=0.0)
            )

        # Muon groups for transformer matrices (grouped by shape)
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

    def warm_up_finished(self):
        """Called after warmup to enable Langevin noise learning if configured."""
        if self.config.langevin_dynamics_noise_learnable:
            self.langevin_dynamics_noise_std.requires_grad = True

    @torch.inference_mode()
    def generate(
        self,
        tokens: List[int],
        max_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        seed: int = 42,
        mcmc_steps_override: Optional[int] = None,
    ):
        """
        Generate tokens autoregressively using MCMC inference.

        Args:
            tokens: Initial token IDs (list)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling (None for full distribution)
            seed: Random seed
            mcmc_steps_override: Override number of MCMC steps for inference

        Yields:
            Generated token IDs
        """
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)

        ids = torch.tensor([tokens], dtype=torch.long, device=device)

        # Override MCMC steps if specified
        orig_mcmc_steps = self.config.mcmc_num_steps
        if mcmc_steps_override is not None:
            self.config.mcmc_num_steps = mcmc_steps_override

        try:
            for _ in range(max_tokens):
                # Run MCMC to get predicted distribution for next token
                with torch.enable_grad():  # MCMC needs gradients
                    predicted_distributions, _ = self(ids, learning=False, no_randomness=True)

                # Get final distribution (B*S, V) -> last position
                logits = predicted_distributions[-1]  # (B*S, V)
                logits = logits.view(1, -1, self.config.vocab_size)  # (1, S, V)
                logits = logits[:, -1, :]  # (1, V) - last position

                # Top-k filtering
                if top_k is not None and top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')

                # Temperature and sampling
                if temperature > 0:
                    logits = logits / temperature
                    probs = F.softmax(logits, dim=-1)
                    next_id = torch.multinomial(probs, num_samples=1, generator=rng)
                else:
                    next_id = torch.argmax(logits, dim=-1, keepdim=True)

                ids = torch.cat([ids, next_id], dim=1)
                yield next_id.item()
        finally:
            # Restore original MCMC steps
            self.config.mcmc_num_steps = orig_mcmc_steps
