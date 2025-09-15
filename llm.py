import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import math
import random
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
import time
from transformers import AutoTokenizer
from dataclasses import dataclass
from typing import List, Optional, Tuple
import warnings
import os
import pickle
import json
from torchtune.modules import RotaryPositionalEmbeddings
warnings.filterwarnings('ignore')

def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🌱 Set all seeds to {seed}")

@dataclass
class MoEModelConfig:
    # Model architecture
    d_model: int = 384
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1536
    batch_size: int = 24
    max_steps: int = 1000

    # Training parameters
    gradient_accumulation_steps: int = 4
    muon_lr: float = 0.01

    # Data parameters
    max_seq_len: int = 512
    num_documents: int = 2000
    max_tokens: int = 500000

    # Evaluation
    eval_every: int = 500
    eval_steps: int = 100
    top_k_list: Tuple[int, ...] = (1, 5)
    ece_num_bins: int = 15
    token_loss_hist_bins: int = 30
    metrics_dir: str = "metrics"

    # Regularization
    weight_decay: float = 0.1
    dropout: float = 0.1
    grad_clip: float = 1.0

    # Technical
    use_amp: bool = True
    vocab_size: Optional[int] = None
    log_milestones: Tuple[int, ...] = (2000, 5000, 10000)

    # MoE specific parameters
    num_experts: int = 8
    expert_top_k: int = 2
    load_balancing_weight: float = 0.01

    def __post_init__(self):
        self.d_k = self.d_model // self.n_heads
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"

@torch.compile
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Newton-Schulz iteration to compute the zeroth power / orthogonalization of G."""
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()

    if G.size(-2) > G.size(-1):
        X = X.mT

    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)

    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT

    return X

class Muon(torch.optim.Optimizer):
    """Muon - MomentUm Orthogonalized by Newton-schulz"""
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad
                state = self.state[p]

                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)

                buf = state["momentum_buffer"]
                buf.lerp_(g, 1 - group["momentum"])
                g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])
                p.add_(g.view_as(p), alpha=-group["lr"] * max(1, p.size(-2) / p.size(-1))**0.5)
	
def load_and_cache_data(config: MoEModelConfig, cache_dir: str = "data_cache"):
    """Load and cache tokenized data to avoid reprocessing"""
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = f"{cache_dir}/tokenized_data_{config.num_documents}_{config.max_tokens}.pkl"

    # Check if cached data exists
    if os.path.exists(cache_file):
        print(f"📦 Loading cached data from {cache_file}")
        with open(cache_file, 'rb') as f:
            cached_data = pickle.load(f)

        texts = cached_data['texts']
        tokenizer = cached_data['tokenizer']
        tokens = cached_data['tokens']
        config.vocab_size = tokenizer.vocab_size

        print(f"✅ Loaded {len(texts)} documents, {len(tokens):,} tokens from cache")
        return texts, tokenizer, tokens

    print(f"🔄 Processing new data (will cache for future use)")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M", token=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    dataset = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", split="train", streaming=True, token=False)

    texts = []
    for i, item in enumerate(dataset):
        if i >= config.num_documents:
            break
        texts.append(item["text"][:3000])

    print(f"Loaded {len(texts)} documents")

    # Tokenize
    print("Tokenizing texts...")
    all_tokens = []
    for text in tqdm(texts, desc="Tokenizing"):
        tokens = tokenizer.encode(text, add_special_tokens=False)
        all_tokens.extend(tokens)

    tokens = all_tokens[:config.max_tokens]
    print(f"Using {len(tokens):,} tokens")
    config.vocab_size = tokenizer.vocab_size

    # Cache the processed data
    cached_data = {'texts': texts, 'tokenizer': tokenizer, 'tokens': tokens}
    with open(cache_file, 'wb') as f:
        pickle.dump(cached_data, f)

    print(f"💾 Cached data to {cache_file}")
    return texts, tokenizer, tokens

class TextTokenDataset(Dataset):
    def __init__(self, tokens: List[int], seq_len: int = 512):
        self.tokens = tokens
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.tokens) - self.seq_len)

    def __getitem__(self, idx):
        x = torch.tensor(self.tokens[idx:idx + self.seq_len], dtype=torch.long)
        y = torch.tensor(self.tokens[idx + 1:idx + self.seq_len + 1], dtype=torch.long)
        return x, y

class Rotary(nn.Module):
    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        self.rope = RotaryPositionalEmbeddings(dim=dim, max_seq_len=max_seq_len, base=10000)

    def forward(self, x_BTHD: torch.Tensor):
        # x_BTHD shape: [B, T, H, D] - need to convert to [B, T, H, D] for torchtune
        # torchtune expects [batch, seq_len, num_heads, head_dim]
        # Our input is already [B, T, H, D] which matches torchtune's expectation
        return self.rope(x_BTHD)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.rotary = Rotary(self.d_k, max_seq_len)
        self.dropout = dropout

    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)
        # B, T = x.size(0), x.size(1)
        # qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_k).permute(2, 0, 3, 1, 4)
        # Q, K, V = qkv[0], qkv[1], qkv[2]  # [B, H, T, D]

        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.n_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        Q, K, V = qkv[0], qkv[1], qkv[2] # [B, H, T, D]

        # Q = self.rotary(Q)
        # K = self.rotary(K)
        # Apply RoPE on [B, T, H, D]
        Q = self.rotary(Q.transpose(1, 2)).transpose(1, 2)
        K = self.rotary(K.transpose(1, 2)).transpose(1, 2)

        attn_output = F.scaled_dot_product_attention(
            Q, K, V, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        # attn_output = attn_output.transpose(1, 2).reshape(B, T, self.d_model)
        return self.w_o(attn_output)



class Expert(nn.Module):
    """Single expert network (essentially a FeedForward layer)"""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff, bias=False)
        self.linear2 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.silu(self.linear1(x))))

class TopKRouter(nn.Module):
    """Router that selects top-k experts for each token"""
    def __init__(self, d_model: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        self.noise_std = 0.1  # Standard deviation for noise during training

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor [batch_size, seq_len, d_model]

        Returns:
            - router_weights: Softmax weights for selected experts [batch_size, seq_len, top_k]
            - expert_indices: Indices of selected experts [batch_size, seq_len, top_k]
            - router_probs: Full probability distribution over experts (for load balancing loss)
        """
        batch_size, seq_len, d_model = x.shape

        # Compute router logits
        router_logits = self.gate(x)  # [batch_size, seq_len, num_experts]

        # Add noise during training for exploration
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(router_logits) * self.noise_std
            router_logits = router_logits + noise

        # Get full probability distribution (for load balancing loss)
        router_probs = F.softmax(router_logits, dim=-1)

        # Select top-k experts
        top_k_logits, top_k_indices = torch.topk(router_logits, self.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_logits, dim=-1)

        return top_k_weights, top_k_indices, router_probs

class MixtureOfExperts(nn.Module):
    """Mixture of Experts layer with top-k routing"""
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int = 8,
        top_k: int = 2,
        dropout: float = 0.1,
        load_balancing_weight: float = 0.01
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.load_balancing_weight = load_balancing_weight

        # Create experts
        self.experts = nn.ModuleList([
            Expert(d_model, d_ff, dropout) for _ in range(num_experts)
        ])

        # Create router
        self.router = TopKRouter(d_model, num_experts, top_k)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: Input tensor [batch_size, seq_len, d_model]

        Returns:
            - output: MoE output [batch_size, seq_len, d_model]
            - aux_loss: Load balancing auxiliary loss (only during training)
        """
        batch_size, seq_len, d_model = x.shape

        # Get routing decisions
        router_weights, expert_indices, router_probs = self.router(x)

        # Initialize output tensor
        output = torch.zeros_like(x)

        # Process each expert
        for expert_idx in range(self.num_experts):
            # Find tokens routed to this expert
            expert_mask = (expert_indices == expert_idx).any(dim=-1)  # [batch_size, seq_len]

            if expert_mask.any():
                # Get tokens for this expert
                expert_input = x[expert_mask]  # [num_tokens, d_model]

                # Apply expert
                expert_output = self.experts[expert_idx](expert_input)

                # Get weights for this expert - CORRECTED APPROACH
                # First get the mask for this expert's positions
                mask_for_expert = (expert_indices == expert_idx)  # [batch, seq, top_k]
                # Find which position (0 or 1) this expert appears in for relevant tokens
                positions = mask_for_expert[expert_mask].float().argmax(dim=-1)
                # Gather weights only for relevant tokens
                expert_weights = router_weights[expert_mask].gather(
                    -1, positions.unsqueeze(-1)
                ).squeeze(-1)

                # Add weighted expert output to result
                output[expert_mask] += expert_weights.unsqueeze(-1) * expert_output

        # Compute load balancing loss during training
        aux_loss = None
        if self.training:
            aux_loss = self._compute_load_balancing_loss(router_probs, expert_indices)

        return output, aux_loss

    def _compute_load_balancing_loss(
        self,
        router_probs: torch.Tensor,
        expert_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute auxiliary loss to ensure balanced expert usage.
        This encourages the router to distribute tokens evenly across experts.
        """
        # Compute the fraction of tokens routed to each expert
        expert_mask = F.one_hot(expert_indices, num_classes=self.num_experts).float()
        tokens_per_expert = expert_mask.sum(dim=[0, 1, 2]) / expert_mask.sum()

        # Compute the average probability of routing to each expert
        router_prob_mean = router_probs.mean(dim=[0, 1])

        # Load balancing loss encourages uniform distribution
        aux_loss = torch.sum(tokens_per_expert * router_prob_mean) * self.num_experts

        return aux_loss * self.load_balancing_weight

class MoETransformerBlock(nn.Module):
    """Transformer block with MoE"""
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        max_seq_len: int,
        num_experts: int = 8,
        top_k: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()

        # Attention layer
        self.attention = MultiHeadAttention(d_model, n_heads, max_seq_len, dropout)

        # MoE layer
        self.feed_forward = MixtureOfExperts(
            d_model, d_ff, num_experts, top_k, dropout
        )

        # Normalization layers
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Self-attention
        attn_out = self.attention(self.norm1(x))
        x = x + self.dropout(attn_out)

        # MoE feed-forward
        ff_out, aux_loss = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_out)
        return x, aux_loss


class MoEMinimalLLM(nn.Module):
    """Minimal LLM with Mixture of Experts"""
    def __init__(self, config: MoEModelConfig):
        super().__init__()
        self.config = config

        # Token embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_dropout = nn.Dropout(config.dropout)

        # Transformer blocks with MoE
        self.transformer_blocks = nn.ModuleList([
            MoETransformerBlock(
                config.d_model,
                config.n_heads,
                config.d_ff,
                config.max_seq_len,
                config.num_experts,
                config.expert_top_k,
                config.dropout
            )
            for i in range(config.n_layers)
        ])

        # Output layers
        self.norm = nn.RMSNorm(config.d_model)
        self.output_dropout = nn.Dropout(config.dropout)

        # Language modeling head (tied with embeddings)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, return_aux_loss=True):
        # Token embeddings
        x = self.token_embedding(x) * math.sqrt(self.config.d_model)
        x = self.position_dropout(x)

        # Collect auxiliary losses from MoE layers
        aux_losses = []

        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x, aux_loss = block(x)
            if aux_loss is not None and return_aux_loss:
                aux_losses.append(aux_loss)

        # Output projection
        x = self.norm(x)
        x = self.output_dropout(x)
        logits = self.lm_head(x)

        # Combine auxiliary losses
        total_aux_loss = sum(aux_losses) if aux_losses else None

        if return_aux_loss:
            return logits, total_aux_loss
        return logits

def _init_eval_accumulators(config: MoEModelConfig):
    """Create accumulators for streaming eval metrics."""
    top_k_list = list(getattr(config, 'top_k_list', (1, 5)))
    ece_bins = getattr(config, 'ece_num_bins', 15)
    token_hist_bins = getattr(config, 'token_loss_hist_bins', 30)

    return {
        'total_loss_sum': 0.0,
        'total_token_count': 0,
        'topk_correct_counts': {int(k): 0 for k in top_k_list},
        'total_correct_top1': 0,
        'entropy_sum': 0.0,
        'repetition_equal_count': 0,   # count of positions where pred_t == pred_{t-1}
        'repetition_total_positions': 0,
        'ece_bin_edges': torch.linspace(0.0, 1.0, steps=ece_bins + 1).cpu(),
        'ece_bin_counts': torch.zeros(ece_bins, dtype=torch.long).cpu(),
        'ece_conf_sums': torch.zeros(ece_bins, dtype=torch.double).cpu(),
        'ece_correct_sums': torch.zeros(ece_bins, dtype=torch.long).cpu(),
        'loss_hist_edges': torch.linspace(0.0, 10.0, steps=token_hist_bins + 1).cpu(),
        'loss_hist_counts': torch.zeros(token_hist_bins, dtype=torch.long).cpu(),
    }

def _accumulate_batch_metrics(
    accumulators: dict,
    logits: torch.Tensor,
    targets: torch.Tensor,
    top_k_list: List[int],
    use_amp: bool
) -> None:
    """Accumulate metrics for a single batch. logits, targets shapes [B, T, V] and [B, T]."""
    batch_size, seq_len, vocab_size = logits.shape
    device = logits.device

    # Cross-entropy per-token
    per_token_loss = F.cross_entropy(
        logits.view(-1, vocab_size), targets.view(-1), reduction='none'
    )  # [B*T]
    accumulators['total_loss_sum'] += float(per_token_loss.sum().detach().cpu())
    accumulators['total_token_count'] += int(per_token_loss.numel())

    # Token-wise loss histogram (streaming)
    # Bucketize on CPU to keep memory in check
    loss_cpu = per_token_loss.detach().cpu()
    bin_idx = torch.bucketize(loss_cpu, accumulators['loss_hist_edges'], right=False) - 1
    bin_idx = bin_idx.clamp(min=0, max=accumulators['loss_hist_counts'].numel() - 1)
    accumulators['loss_hist_counts'] += torch.bincount(
        bin_idx, minlength=accumulators['loss_hist_counts'].numel()
    )

    # Predictions and correctness
    preds = logits.argmax(dim=-1)  # [B, T]
    correct_top1 = (preds == targets).sum().item()
    accumulators['total_correct_top1'] += int(correct_top1)

    # Top-k accuracy
    flat_targets = targets.view(-1)
    for k in top_k_list:
        k_safe = int(min(int(k), vocab_size))
        if k_safe <= 0:
            continue
        topk = torch.topk(logits.view(-1, vocab_size), k_safe, dim=-1).indices  # [B*T, k]
        in_topk = (topk == flat_targets.unsqueeze(-1)).any(dim=-1)
        accumulators['topk_correct_counts'][int(k)] += int(in_topk.sum().item())

    # Predictive entropy
    # Use log-sum-exp trick via log_softmax to improve stability
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1)  # [B, T]
    accumulators['entropy_sum'] += float(entropy.sum().detach().cpu())

    # ECE accumulators based on predicted confidence and correctness
    pred_conf, pred_class = probs.max(dim=-1)  # [B, T]
    correct = (pred_class == targets).detach().cpu().view(-1)
    conf = pred_conf.detach().cpu().view(-1)
    ece_edges = accumulators['ece_bin_edges']
    ece_bins = ece_edges.numel() - 1
    conf_bin_idx = torch.bucketize(conf, ece_edges, right=True) - 1
    conf_bin_idx = conf_bin_idx.clamp(min=0, max=ece_bins - 1)

    # Aggregate per bin
    accumulators['ece_bin_counts'] += torch.bincount(conf_bin_idx, minlength=ece_bins)
    accumulators['ece_conf_sums'] += torch.bincount(
        conf_bin_idx, weights=conf.to(torch.double), minlength=ece_bins
    )
    accumulators['ece_correct_sums'] += torch.bincount(
        conf_bin_idx, weights=correct.to(torch.long), minlength=ece_bins
    )

    # Repetition rate across predictions within each sequence
    if seq_len > 1:
        repeats = (preds[:, 1:] == preds[:, :-1]).sum().item()
        accumulators['repetition_equal_count'] += int(repeats)
        accumulators['repetition_total_positions'] += int(batch_size * (seq_len - 1))

def _finalize_eval_metrics(accumulators: dict, config: MoEModelConfig) -> dict:
    """Compute final metrics from accumulators."""
    total_tokens = max(1, accumulators['total_token_count'])
    avg_loss = accumulators['total_loss_sum'] / total_tokens
    perplexity = math.exp(min(avg_loss, 20))
    accuracy_top1 = accumulators['total_correct_top1'] / total_tokens

    # Top-k accuracies
    topk_acc = {
        f'top{k}': accumulators['topk_correct_counts'][k] / total_tokens
        for k in sorted(accumulators['topk_correct_counts'].keys())
    }

    # Mean predictive entropy
    mean_entropy = accumulators['entropy_sum'] / total_tokens

    # ECE computation
    bin_counts = accumulators['ece_bin_counts'].to(torch.double)
    nonzero = bin_counts > 0
    avg_conf = torch.zeros_like(bin_counts)
    avg_acc = torch.zeros_like(bin_counts)
    avg_conf[nonzero] = (accumulators['ece_conf_sums'][nonzero] / bin_counts[nonzero])
    avg_acc[nonzero] = (accumulators['ece_correct_sums'][nonzero].to(torch.double) / bin_counts[nonzero])
    weights = bin_counts / max(1.0, float(bin_counts.sum().item()))
    ece = float((weights * (avg_acc - avg_conf).abs()).sum().item())

    # Repetition rate
    if accumulators['repetition_total_positions'] > 0:
        repetition_rate = accumulators['repetition_equal_count'] / accumulators['repetition_total_positions']
    else:
        repetition_rate = 0.0

    # Token-wise loss histogram
    loss_hist = {
        'bin_edges': accumulators['loss_hist_edges'].tolist(),
        'counts': accumulators['loss_hist_counts'].tolist()
    }

    metrics = {
        'val_loss': avg_loss,
        'val_accuracy': accuracy_top1,
        'val_perplexity': perplexity,
        'topk_accuracy': topk_acc,
        'predictive_entropy': mean_entropy,
        'ece': ece,
        'repetition_rate': repetition_rate,
        'token_loss_histogram': loss_hist,
    }
    return metrics

def evaluate_model(
    model: nn.Module,
    val_loader: DataLoader,
    config: MoEModelConfig,
    save_path: Optional[str] = None
):
    """Evaluate model performance with extended metrics and optional JSON save."""
    model.eval()
    device = next(model.parameters()).device

    accum = _init_eval_accumulators(config)
    top_k_list = list(getattr(config, 'top_k_list', (1, 5)))

    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            if i >= config.eval_steps:
                break
            x, y = x.to(device), y.to(device)

            with autocast(enabled=config.use_amp):
                logits = model(x, return_aux_loss=False)

            _accumulate_batch_metrics(accum, logits, y, top_k_list, config.use_amp)

    metrics = _finalize_eval_metrics(accum, config)

    # Optional save
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(metrics, f, indent=2)

    model.train()
    return metrics

def _evaluate_on_texts(
    model: nn.Module,
    tokenizer,
    texts: List[str],
    config: MoEModelConfig
) -> dict:
    """Teacher-forced evaluation on raw texts for sanity checks."""
    device = next(model.parameters()).device
    accum = _init_eval_accumulators(config)
    top_k_list = list(getattr(config, 'top_k_list', (1, 5)))

    model.eval()
    with torch.no_grad():
        for text in texts:
            ids = tokenizer.encode(text, add_special_tokens=False)
            if len(ids) < 2:
                continue
            # Create single batch sequence [1, T-1] input and targets
            x = torch.tensor(ids[:-1], dtype=torch.long, device=device).unsqueeze(0)
            y = torch.tensor(ids[1:], dtype=torch.long, device=device).unsqueeze(0)

            with autocast(enabled=config.use_amp):
                logits = model(x, return_aux_loss=False)

            _accumulate_batch_metrics(accum, logits, y, top_k_list, config.use_amp)

    metrics = _finalize_eval_metrics(accum, config)
    model.train()
    return metrics

def _sample_generate(
    model: nn.Module,
    tokenizer,
    device: torch.device,
    prompt: str,
    max_length: int = 50,
    temperature: float = 0.8,
    top_k: int = 50
) -> str:
    """Simple sampler for sanity check generations."""
    model.eval()
    input_ids = torch.tensor(tokenizer.encode(prompt, return_tensors=None, add_special_tokens=False), dtype=torch.long, device=device).unsqueeze(0)
    generated = input_ids.clone()
    with torch.no_grad():
        for _ in range(max_length):
            logits = model(generated, return_aux_loss=False)
            next_logits = logits[0, -1] / max(1e-6, temperature)
            if top_k > 0:
                topk_vals, topk_idx = torch.topk(next_logits, k=min(top_k, next_logits.size(-1)))
                mask = next_logits < topk_vals[-1]
                next_logits = next_logits.masked_fill(mask, float('-inf'))
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token.view(1, 1)], dim=1)
    text = tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)
    # Return only the newly generated part
    return text[len(prompt):]

def run_sanity_checks(
    model: nn.Module,
    tokenizer,
    config: MoEModelConfig,
    out_dir: str
) -> dict:
    """Run simple sanity checks on classic prompts and save results."""
    os.makedirs(out_dir, exist_ok=True)
    classic_texts = [
        "the quick brown fox jumps over the lazy dog",
        "to be or not to be, that is the question",
        "hello world!",
        "once upon a time there was a",
        "1 1 2 3 5 8 13 21"
    ]
    # Teacher-forced evaluation on these texts (aggregated)
    agg_metrics = _evaluate_on_texts(model, tokenizer, classic_texts, config)

    # Per-text metrics
    per_text = {}
    for t in classic_texts:
        per_text[t] = _evaluate_on_texts(model, tokenizer, [t], config)

    # Generations for quick qualitative check
    device = next(model.parameters()).device
    gen_prompts = [
        "The quick brown fox",
        "To be or not to be",
        "Hello world",
        "Once upon a time",
        "In a galaxy far, far away"
    ]
    generations = []
    for p in gen_prompts:
        gen = _sample_generate(model, tokenizer, device, p, max_length=40, temperature=0.8, top_k=50)
        generations.append({'prompt': p, 'generated': gen})

    sanity = {
        'aggregated_metrics': agg_metrics,
        'per_text_metrics': per_text,
        'generations': generations
    }

    with open(os.path.join(out_dir, 'sanity_checks.json'), 'w') as f:
        json.dump(sanity, f, indent=2)

    return sanity

def setup_muon_optimizer(model: nn.Module, config: MoEModelConfig):
    """Setup Muon optimizer with hybrid approach"""
    muon_params = []
    adamw_params = []

    for name, param in model.named_parameters():
        if (param.ndim == 2 and 
            'token_embedding' not in name and 
            'norm' not in name and 
            param.requires_grad):
            muon_params.append(param)
        else:
            adamw_params.append(param)

    print(f"  Muon parameters: {sum(p.numel() for p in muon_params):,}")
    print(f"  AdamW parameters: {sum(p.numel() for p in adamw_params):,}")

    muon_optimizer = Muon(muon_params, lr=config.muon_lr, momentum=0.95)
    adamw_optimizer = torch.optim.AdamW(adamw_params, lr=config.muon_lr*0.1, weight_decay=config.weight_decay)

    return [muon_optimizer, adamw_optimizer]


def train_moe_model(
    config: MoEModelConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    tokenizer=None
):
    """Train the MoE model"""
    print(f"\n🚀 Training MoE model with {config.num_experts} experts (top-{config.expert_top_k})")

    # Initialize model
    set_seed(42)
    model = MoEMinimalLLM(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    active_params = sum(p.numel() for n, p in model.named_parameters()
                       if 'expert' not in n)
    expert_params = total_params - active_params

    print(f"  📊 Total parameters: {total_params:,}")
    print(f"  📊 Active parameters: {active_params:,}")
    print(f"  📊 Expert parameters: {expert_params:,}")
    print(f"  📊 Parameter efficiency: {active_params/total_params:.1%} active per forward pass")

    # Setup optimizers
    optimizers = setup_muon_optimizer(model, config)

    # Learning rate schedule
    schedulers = []
    for optimizer in optimizers:
        warmup_steps = config.max_steps // 20
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                progress = (step - warmup_steps) / (config.max_steps - warmup_steps)
                return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        schedulers.append(scheduler)

    scaler = GradScaler() if config.use_amp else None

    # Prepare metrics directory
    metrics_dir = getattr(config, 'metrics_dir', 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)

    # Training loop
    model.train()
    step = 0
    pbar = tqdm(total=config.max_steps, desc="Training MoE")

    while step < config.max_steps:
        for batch_idx, (x, y) in enumerate(train_loader):
            if step >= config.max_steps:
                break

            x, y = x.to(device), y.to(device)

            # Forward pass
            if config.use_amp:
                with autocast():
                    logits, aux_loss = model(x, return_aux_loss=True)
                    ce_loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))

                    # Combine main loss and auxiliary loss
                    total_loss = ce_loss
                    if aux_loss is not None:
                        total_loss = total_loss + aux_loss

                    loss = total_loss / config.gradient_accumulation_steps
                scaler.scale(loss).backward()
            else:
                logits, aux_loss = model(x, return_aux_loss=True)
                ce_loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))

                total_loss = ce_loss
                if aux_loss is not None:
                    total_loss = total_loss + aux_loss

                loss = total_loss / config.gradient_accumulation_steps
                loss.backward()

            # Optimizer step
            if (step + 1) % config.gradient_accumulation_steps == 0:
                if config.use_amp:
                    for optimizer in optimizers:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

                    for optimizer in optimizers:
                        scaler.step(optimizer)
                        optimizer.zero_grad()
                    for scheduler in schedulers:
                        scheduler.step()
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                    for optimizer in optimizers:
                        optimizer.step()
                        optimizer.zero_grad()
                    for scheduler in schedulers:
                        scheduler.step()

            # Logging
            if step % 100 == 0:
                with torch.no_grad():
                    predictions = logits.argmax(dim=-1)
                    accuracy = (predictions == y).float().mean().item()
                    current_loss = ce_loss.item()
                    perplexity = math.exp(min(current_loss, 20))

                pbar.set_postfix({
                    'loss': f'{current_loss:.4f}',
                    'aux': f'{aux_loss.item() if aux_loss is not None else 0:.4f}',
                    'acc': f'{accuracy:.3f}',
                    'ppl': f'{perplexity:.1f}'
                })

            # Evaluation
            if step % config.eval_every == 0 and step > 0:
                eval_path = os.path.join(metrics_dir, f"eval_step_{step}.json")
                eval_metrics = evaluate_model(model, val_loader, config, save_path=eval_path)
                topk_str = ", ".join([f"top{k}:{eval_metrics['topk_accuracy'].get(f'top{k}', 0):.3f}" for k in getattr(config, 'top_k_list', (1, 5))])
                print(f"\nStep {step}: Val Loss: {eval_metrics['val_loss']:.4f}, "
                      f"Val Acc@1: {eval_metrics['val_accuracy']:.4f}, "
                      f"{topk_str}, PPL: {eval_metrics['val_perplexity']:.2f}, "
                      f"ECE: {eval_metrics['ece']:.4f}, Entropy: {eval_metrics['predictive_entropy']:.3f}, "
                      f"Repeat: {eval_metrics['repetition_rate']:.3f}")

            # Milestone evaluations
            if step in getattr(config, 'log_milestones', ()):    
                milestone_path = os.path.join(metrics_dir, f"milestone_{step}.json")
                eval_metrics = evaluate_model(model, val_loader, config, save_path=milestone_path)
                print(f"\n🧪 Milestone {step}: Val Loss: {eval_metrics['val_loss']:.4f}, ECE: {eval_metrics['ece']:.4f}")

            step += 1
            if step % 20 == 0:
                pbar.update(20)

    pbar.close()

    # Final evaluation
    final_eval_path = os.path.join(metrics_dir, 'final_eval.json')
    final_eval = evaluate_model(model, val_loader, config, save_path=final_eval_path)
    print(f"\n📊 Final Results:")
    print(f"   Val Loss: {final_eval['val_loss']:.4f}")
    print(f"   Val Accuracy@1: {final_eval['val_accuracy']:.4f}")
    print(f"   Val Perplexity: {final_eval['val_perplexity']:.2f}")
    print(f"   ECE: {final_eval['ece']:.4f}, Entropy: {final_eval['predictive_entropy']:.3f}, Repeat: {final_eval['repetition_rate']:.3f}")

    sanity_metrics = None
    if tokenizer is not None:
        try:
            sanity_dir = os.path.join(metrics_dir, 'sanity')
            sanity_metrics = run_sanity_checks(model, tokenizer, config, sanity_dir)
            print("\n🧪 Wrote sanity check metrics and generations to metrics/sanity/sanity_checks.json")
        except Exception as e:
            print(f"⚠️  Sanity checks failed: {e}")

    return model, final_eval, sanity_metrics

if __name__ == "__main__":
    # Check system
    print(f"🔍 Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Set seed
    set_seed(42)

    # Load data first to get vocab_size
    temp_config = MoEModelConfig()  # Use MoE config for data loading
    texts, tokenizer, tokens = load_and_cache_data(temp_config)
    vocab_size = temp_config.vocab_size

    # Use MoE config and set vocab_size
    config = MoEModelConfig(vocab_size=vocab_size)

    dataset = TextTokenDataset(tokens, config.max_seq_len)

    # Train/val split
    val_size = len(dataset) // 10
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2)

    print(f"📊 Dataset: {len(train_dataset)} train, {len(val_dataset)} val samples")

    # Train MoE model
    print(f"\n{'='*60}")
    print(f"🧪 TRAINING: Mixture of Experts Model")
    print(f"{'='*60}")

    print(f"\n📋 MoE Model Configuration:")
    print(f"   Architecture: {config.d_model}d, {config.n_layers}L, {config.n_heads}H, {config.d_ff}ff")
    print(f"   MoE: {config.num_experts} experts, top-{config.expert_top_k} routing")
    print(f"   Training: {config.max_steps} steps, batch size {config.batch_size}")
    print(f"   Data: {config.max_tokens:,} tokens, seq_len {config.max_seq_len}")

    # Train model
    start_time = time.time()
    model, final_metrics, _ = train_moe_model(config, train_loader, val_loader, tokenizer=tokenizer)
    total_time = time.time() - start_time

    print(f"\n🎯 MoE Model Results:")
    print(f"⏱️ Training time: {total_time/60:.1f} minutes")
    print(f"🏆 Final Results:")
    print(f"   Validation Loss: {final_metrics['val_loss']:.4f}")
    print(f"   Validation Accuracy: {final_metrics['val_accuracy']:.4f}")
    print(f"   Validation Perplexity: {final_metrics['val_perplexity']:.2f}")
    print(f"{'='*60}")