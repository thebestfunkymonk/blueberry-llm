"""
Model evaluation functions.

This module provides utilities for evaluating model performance
including loss computation, perplexity, and accuracy metrics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast
import math
from typing import Dict, Any, Optional, Tuple, List
from configs import AdaptiveMoEModelConfig


def evaluate_model(
    model: nn.Module,
    val_loader: DataLoader,
    config: AdaptiveMoEModelConfig,
    max_eval_steps: Optional[int] = None
) -> Dict[str, Any]:
    """
    Evaluate model performance on validation data.
    
    Args:
        model: Model to evaluate
        val_loader: Validation data loader
        config: Model configuration
        max_eval_steps: Maximum number of evaluation steps (None for full evaluation)
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    total_correct = 0
    # For top-k accuracies
    topk_ks: Tuple[int, ...] = getattr(config, 'eval_top_k', (1, 5))
    topk_correct = {k: 0 for k in topk_ks}
    # For repetition rate (based on predictions)
    total_repetition_events = 0
    total_repetition_candidates = 0
    # For predictive entropy and ECE
    sum_entropy = 0.0
    # ECE accumulators
    num_bins = getattr(config, 'ece_num_bins', 15)
    ece_bin_counts = [0] * num_bins
    ece_conf_sums = [0.0] * num_bins
    ece_acc_sums = [0.0] * num_bins
    # Loss histogram accumulators
    loss_hist_bins = getattr(config, 'loss_hist_num_bins', 30)
    loss_hist_min = 0.0
    loss_hist_max = 10.0
    loss_hist_counts = [0] * loss_hist_bins
    num_steps = 0
    
    device = next(model.parameters()).device
    eval_steps = max_eval_steps or config.eval_steps

    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            if i >= eval_steps:
                break
                
            x, y = x.to(device), y.to(device)
            num_steps += 1

            with autocast('cuda', enabled=config.use_amp):
                # Handle MoE models that return aux loss
                if hasattr(model, 'forward') and 'return_aux_loss' in model.forward.__code__.co_varnames:
                    logits = model(x, return_aux_loss=False)
                else:
                    logits = model(x)
                
                # Compute token-wise cross-entropy loss and aggregate
                log_probs = F.log_softmax(logits, dim=-1)
                nll_tokens = -log_probs.view(-1, config.vocab_size).gather(dim=-1, index=y.view(-1).unsqueeze(-1)).squeeze(-1)
                loss = nll_tokens.sum()

            # Accumulate metrics
            batch_tokens = y.numel()
            total_loss += loss.item()
            total_tokens += batch_tokens

            # Predictions and probabilities
            predictions = logits.argmax(dim=-1)
            probs = F.softmax(logits, dim=-1)

            # Top-k accuracies
            if topk_ks:
                flat_logits = logits.view(-1, logits.size(-1))
                flat_targets = y.view(-1)
                max_k = max(topk_ks)
                topk_indices = flat_logits.topk(k=max_k, dim=-1).indices
                for k in topk_ks:
                    correct_k = (topk_indices[:, :k] == flat_targets.unsqueeze(-1)).any(dim=-1).sum().item()
                    topk_correct[k] += int(correct_k)

            # Top-1 accuracy for backward compatibility
            total_correct += (predictions == y).sum().item()

            # Repetition rate (predicted token equals previous predicted token within each sequence)
            if predictions.size(1) > 1:
                rep_matches = (predictions[:, 1:] == predictions[:, :-1]).sum().item()
                total_repetition_events += int(rep_matches)
                total_repetition_candidates += predictions.size(0) * (predictions.size(1) - 1)

            # Predictive entropy (per-token) and ECE accumulators
            flat_probs = probs.view(-1, probs.size(-1))
            # entropy = -sum p log p
            entropy = -(flat_probs * torch.clamp(flat_probs, min=1e-12).log()).sum(dim=-1)
            sum_entropy += float(entropy.sum().item())

            confidences, pred_flat = flat_probs.max(dim=-1)
            correctness = (pred_flat == y.view(-1))
            # Bin confidences for ECE
            bin_indices = torch.clamp((confidences * num_bins).floor().long(), 0, num_bins - 1)
            for b in range(num_bins):
                mask = (bin_indices == b)
                count_b = int(mask.sum().item())
                if count_b > 0:
                    ece_bin_counts[b] += count_b
                    ece_conf_sums[b] += float(confidences[mask].sum().item())
                    ece_acc_sums[b] += float(correctness[mask].float().sum().item())

            # Loss histogram accumulation
            # Clamp nll range for histogram stability
            nll_clamped = torch.clamp(nll_tokens, min=loss_hist_min, max=loss_hist_max - 1e-6)
            bin_ids = torch.clamp(((nll_clamped - loss_hist_min) / (loss_hist_max - loss_hist_min) * loss_hist_bins).floor().long(), 0, loss_hist_bins - 1)
            for b in range(loss_hist_bins):
                loss_hist_counts[b] += int((bin_ids == b).sum().item())

    # Compute final metrics
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0
    perplexity = compute_perplexity(avg_loss)
    # Top-k accuracies normalized
    topk_metrics = {f'top{k}_accuracy': (topk_correct[k] / total_tokens if total_tokens > 0 else 0.0) for k in topk_ks}
    # Repetition rate
    repetition_rate = (total_repetition_events / total_repetition_candidates) if total_repetition_candidates > 0 else 0.0
    # Predictive entropy
    avg_entropy = (sum_entropy / total_tokens) if total_tokens > 0 else 0.0
    # ECE
    ece = 0.0
    ece_bins: List[Dict[str, float]] = []
    for b in range(num_bins):
        count_b = ece_bin_counts[b]
        if count_b > 0:
            avg_conf_b = ece_conf_sums[b] / count_b
            avg_acc_b = ece_acc_sums[b] / count_b
            ece += abs(avg_acc_b - avg_conf_b) * (count_b / total_tokens)
            ece_bins.append({
                'bin_lower': b / num_bins,
                'bin_upper': (b + 1) / num_bins,
                'count': count_b,
                'avg_confidence': avg_conf_b,
                'avg_accuracy': avg_acc_b
            })
        else:
            ece_bins.append({
                'bin_lower': b / num_bins,
                'bin_upper': (b + 1) / num_bins,
                'count': 0,
                'avg_confidence': 0.0,
                'avg_accuracy': 0.0
            })
    # Loss histogram edges
    loss_hist_edges = [loss_hist_min + (loss_hist_max - loss_hist_min) * i / loss_hist_bins for i in range(loss_hist_bins + 1)]

    model.train()
    
    return {
        'val_loss': avg_loss,
        'val_accuracy': accuracy,
        'val_perplexity': perplexity,
        'total_tokens': total_tokens,
        'num_steps': num_steps,
        **topk_metrics,
        'repetition_rate': repetition_rate,
        'predictive_entropy': avg_entropy,
        'ece': ece,
        'ece_bins': ece_bins,
        'loss_hist_counts': loss_hist_counts,
        'loss_hist_edges': loss_hist_edges,
    }


def compute_perplexity(loss: float, max_perplexity: float = 1000.0) -> float:
    """
    Compute perplexity from cross-entropy loss.
    
    Args:
        loss: Cross-entropy loss value
        max_perplexity: Maximum perplexity to return (for numerical stability)
        
    Returns:
        Perplexity value
    """
    try:
        perplexity = math.exp(min(loss, math.log(max_perplexity)))
        return perplexity
    except (OverflowError, ValueError):
        return max_perplexity


def evaluate_generation_quality(
    model: nn.Module,
    tokenizer,
    prompts: list,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: Optional[int] = None
) -> Dict[str, Any]:
    """
    Evaluate generation quality with sample prompts.
    
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer for encoding/decoding
        prompts: List of prompt strings
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_k: Top-k sampling
        
    Returns:
        Dictionary with generation results
    """
    model.eval()
    device = next(model.parameters()).device
    
    results = {
        'prompts': prompts,
        'generations': [],
        'avg_length': 0.0
    }
    
    with torch.no_grad():
        for prompt in prompts:
            # Encode prompt
            input_ids = torch.tensor(
                tokenizer.encode(prompt), 
                dtype=torch.long, 
                device=device
            ).unsqueeze(0)
            
            # Generate
            if hasattr(model, 'generate'):
                generated = model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k
                )
            else:
                # Simple greedy generation fallback
                generated = simple_generate(
                    model, 
                    input_ids, 
                    max_new_tokens,
                    temperature
                )
            
            # Decode
            generated_text = tokenizer.decode(
                generated[0].cpu().tolist(), 
                skip_special_tokens=True
            )
            
            results['generations'].append({
                'prompt': prompt,
                'generated': generated_text,
                'length': len(generated[0]) - len(input_ids[0])
            })
    
    # Compute average generation length
    if results['generations']:
        results['avg_length'] = sum(
            gen['length'] for gen in results['generations']
        ) / len(results['generations'])
    
    model.train()
    return results


def simple_generate(
    model: nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0
) -> torch.Tensor:
    """
    Simple greedy/sampling generation for models without generate method.
    
    Args:
        model: Model to use for generation
        input_ids: Input token IDs
        max_new_tokens: Maximum new tokens to generate
        temperature: Sampling temperature
        
    Returns:
        Generated token sequence
    """
    for _ in range(max_new_tokens):
        # Get logits
        if hasattr(model, 'forward') and 'return_aux_loss' in model.forward.__code__.co_varnames:
            logits = model(input_ids, return_aux_loss=False)
        else:
            logits = model(input_ids)
        
        # Get next token logits
        next_token_logits = logits[:, -1, :] / temperature
        
        # Sample next token
        if temperature > 0:
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)
        
        # Append to sequence
        input_ids = torch.cat([input_ids, next_token], dim=1)
        
        # Stop if sequence gets too long (prevent memory issues)
        if input_ids.size(1) > 2048:
            break
    
    return input_ids


def compute_model_metrics(
    model: nn.Module,
    val_loader: DataLoader,
    config: AdaptiveMoEModelConfig,
    compute_mfu: bool = False,
    dt: float = 1.0
) -> Dict[str, Any]:
    """
    Compute comprehensive model metrics.
    
    Args:
        model: Model to evaluate
        val_loader: Validation data loader
        config: Model configuration
        compute_mfu: Whether to compute model FLOPs utilization
        dt: Time per iteration (for MFU computation)
        
    Returns:
        Dictionary with all metrics
    """
    # Basic evaluation metrics
    eval_results = evaluate_model(model, val_loader, config)
    
    # Model size metrics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    metrics = {
        **eval_results,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'total_params_M': total_params / 1e6,
        'trainable_params_M': trainable_params / 1e6,
    }
    
    # MoE-specific metrics
    if hasattr(model, 'config') and hasattr(model.config, 'num_experts'):
        active_params = 0
        expert_params = 0
        
        for name, param in model.named_parameters():
            if 'expert' in name:
                expert_params += param.numel()
            else:
                active_params += param.numel()
        
        metrics.update({
            'active_params': active_params,
            'expert_params': expert_params,
            'active_params_M': active_params / 1e6,
            'expert_params_M': expert_params / 1e6,
            'parameter_efficiency': active_params / total_params if total_params > 0 else 0.0,
        })
    
    # Model FLOPs utilization
    if compute_mfu and hasattr(model, 'estimate_mfu'):
        mfu = model.estimate_mfu(fwdbwd_per_iter=1, dt=dt)
        metrics['mfu'] = mfu
        metrics['mfu_percent'] = mfu * 100
    
    return metrics


def benchmark_model_speed(
    model: nn.Module,
    data_loader: DataLoader,
    config: AdaptiveMoEModelConfig,
    num_iterations: int = 10
) -> Dict[str, float]:
    """
    Benchmark model inference speed.
    
    Args:
        model: Model to benchmark
        data_loader: Data loader for benchmark data
        config: Model configuration
        num_iterations: Number of iterations to benchmark
        
    Returns:
        Dictionary with timing metrics
    """
    model.eval()
    device = next(model.parameters()).device
    
    # Warmup
    with torch.no_grad():
        for i, (x, y) in enumerate(data_loader):
            if i >= 3:  # Warmup iterations
                break
            x = x.to(device)
            _ = model(x, return_aux_loss=False) if hasattr(model, 'forward') and 'return_aux_loss' in model.forward.__code__.co_varnames else model(x)
    
    # Benchmark
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    
    if torch.cuda.is_available():
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
    else:
        import time
        start_time = time.time()
    
    total_tokens = 0
    
    with torch.no_grad():
        for i, (x, y) in enumerate(data_loader):
            if i >= num_iterations:
                break
            
            x = x.to(device)
            _ = model(x, return_aux_loss=False) if hasattr(model, 'forward') and 'return_aux_loss' in model.forward.__code__.co_varnames else model(x)
            total_tokens += x.numel()
    
    if torch.cuda.is_available():
        end_event.record()
        torch.cuda.synchronize()
        elapsed_ms = start_event.elapsed_time(end_event)
        elapsed_s = elapsed_ms / 1000
    else:
        elapsed_s = time.time() - start_time
        elapsed_ms = elapsed_s * 1000
    
    model.train()
    
    return {
        'total_time_s': elapsed_s,
        'total_time_ms': elapsed_ms,
        'avg_time_per_iteration_ms': elapsed_ms / num_iterations,
        'tokens_per_second': total_tokens / elapsed_s if elapsed_s > 0 else 0,
        'total_tokens': total_tokens,
        'num_iterations': num_iterations
    }


def evaluate_sanity_check(
    model: nn.Module,
    tokenizer,
    config: AdaptiveMoEModelConfig,
    prompts: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Run simple sanity-check evaluations on classic prompts and report
    token-wise loss histograms and core metrics.
    """
    if prompts is None:
        prompts = [
            "the quick brown fox jumps over the lazy dog",
            "hello world",
            "to be or not to be, that is the question",
        ]
    model.eval()
    device = next(model.parameters()).device

    overall = {
        'total_tokens': 0,
        'sum_loss': 0.0,
        'sum_correct': 0,
        'sum_entropy': 0.0,
        'rep_events': 0,
        'rep_candidates': 0,
        'topk_correct': {k: 0 for k in getattr(config, 'eval_top_k', (1, 5))},
        'ece_bin_counts': [0] * getattr(config, 'ece_num_bins', 15),
        'ece_conf_sums': [0.0] * getattr(config, 'ece_num_bins', 15),
        'ece_acc_sums': [0.0] * getattr(config, 'ece_num_bins', 15),
        'loss_hist_counts': [0] * getattr(config, 'loss_hist_num_bins', 30),
    }
    loss_hist_min = 0.0
    loss_hist_max = 10.0
    loss_hist_bins = getattr(config, 'loss_hist_num_bins', 30)
    num_bins = getattr(config, 'ece_num_bins', 15)
    topk_ks: Tuple[int, ...] = getattr(config, 'eval_top_k', (1, 5))

    per_prompt: List[Dict[str, Any]] = []

    with torch.no_grad():
        for prompt in prompts:
            token_ids = tokenizer.encode(prompt, add_special_tokens=False)
            if len(token_ids) < 2:
                continue
            x_ids = torch.tensor(token_ids[:-1], dtype=torch.long, device=device).unsqueeze(0)
            y_ids = torch.tensor(token_ids[1:], dtype=torch.long, device=device).unsqueeze(0)

            with autocast('cuda', enabled=config.use_amp):
                if hasattr(model, 'forward') and 'return_aux_loss' in model.forward.__code__.co_varnames:
                    logits = model(x_ids, return_aux_loss=False)
                else:
                    logits = model(x_ids)

            # Align shapes
            logits = logits[:, : y_ids.size(1), :]
            log_probs = F.log_softmax(logits, dim=-1)
            probs = F.softmax(logits, dim=-1)

            # Token-wise NLL
            nll = -log_probs.gather(dim=-1, index=y_ids.unsqueeze(-1)).squeeze(-1)
            seq_tokens = int(y_ids.numel())
            sum_loss = float(nll.sum().item())

            # Accumulate overall
            overall['total_tokens'] += seq_tokens
            overall['sum_loss'] += sum_loss

            # Accuracy and repetition
            pred = logits.argmax(dim=-1)
            correct = (pred == y_ids).sum().item()
            overall['sum_correct'] += int(correct)
            if pred.size(1) > 1:
                rep_matches = (pred[:, 1:] == pred[:, :-1]).sum().item()
                overall['rep_events'] += int(rep_matches)
                overall['rep_candidates'] += pred.size(0) * (pred.size(1) - 1)

            # Top-k
            if topk_ks:
                flat_logits = logits.reshape(-1, logits.size(-1))
                flat_targets = y_ids.reshape(-1)
                max_k = max(topk_ks)
                tk = flat_logits.topk(k=max_k, dim=-1).indices
                for k in topk_ks:
                    overall['topk_correct'][k] += int((tk[:, :k] == flat_targets.unsqueeze(-1)).any(dim=-1).sum().item())

            # Entropy and ECE
            flat_probs = probs.reshape(-1, probs.size(-1))
            entropy = -(flat_probs * torch.clamp(flat_probs, min=1e-12).log()).sum(dim=-1)
            overall['sum_entropy'] += float(entropy.sum().item())
            confidences, pred_flat = flat_probs.max(dim=-1)
            correctness = (pred_flat == y_ids.reshape(-1))
            bin_indices = torch.clamp((confidences * num_bins).floor().long(), 0, num_bins - 1)
            for b in range(num_bins):
                mask = (bin_indices == b)
                count_b = int(mask.sum().item())
                if count_b > 0:
                    overall['ece_bin_counts'][b] += count_b
                    overall['ece_conf_sums'][b] += float(confidences[mask].sum().item())
                    overall['ece_acc_sums'][b] += float(correctness[mask].float().sum().item())

            # Loss histogram per prompt
            nll_clamped = torch.clamp(nll, min=loss_hist_min, max=loss_hist_max - 1e-6)
            bin_ids = torch.clamp(((nll_clamped - loss_hist_min) / (loss_hist_max - loss_hist_min) * loss_hist_bins).floor().long(), 0, loss_hist_bins - 1)
            prompt_hist_counts = [0] * loss_hist_bins
            for b in range(loss_hist_bins):
                count_b = int((bin_ids == b).sum().item())
                prompt_hist_counts[b] = count_b
                overall['loss_hist_counts'][b] += count_b

            per_prompt.append({
                'prompt': prompt,
                'num_tokens': seq_tokens,
                'avg_nll': sum_loss / seq_tokens,
                'top1_accuracy': correct / seq_tokens,
                'loss_hist_counts': prompt_hist_counts,
                # Add a short greedy generation sample for qualitative sanity-check
                'generated': tokenizer.decode(
                    (
                        model.generate(x_ids, max_new_tokens=16) if hasattr(model, 'generate') else 
                        simple_generate(model, x_ids, max_new_tokens=16, temperature=0.0)
                    )[0].detach().cpu().tolist(),
                    skip_special_tokens=True
                )
            })

    # Finalize overall metrics
    avg_loss = overall['sum_loss'] / overall['total_tokens'] if overall['total_tokens'] > 0 else float('inf')
    topk = {f'top{k}_accuracy': (overall['topk_correct'][k] / overall['total_tokens'] if overall['total_tokens'] > 0 else 0.0) for k in topk_ks}
    repetition_rate = (overall['rep_events'] / overall['rep_candidates']) if overall['rep_candidates'] > 0 else 0.0
    avg_entropy = (overall['sum_entropy'] / overall['total_tokens']) if overall['total_tokens'] > 0 else 0.0
    ece = 0.0
    ece_bins_out: List[Dict[str, float]] = []
    for b in range(num_bins):
        count_b = overall['ece_bin_counts'][b]
        if count_b > 0:
            avg_conf_b = overall['ece_conf_sums'][b] / count_b
            avg_acc_b = overall['ece_acc_sums'][b] / count_b
            ece += abs(avg_acc_b - avg_conf_b) * (count_b / max(1, overall['total_tokens']))
            ece_bins_out.append({
                'bin_lower': b / num_bins,
                'bin_upper': (b + 1) / num_bins,
                'count': count_b,
                'avg_confidence': avg_conf_b,
                'avg_accuracy': avg_acc_b
            })
        else:
            ece_bins_out.append({
                'bin_lower': b / num_bins,
                'bin_upper': (b + 1) / num_bins,
                'count': 0,
                'avg_confidence': 0.0,
                'avg_accuracy': 0.0
            })

    loss_hist_edges = [loss_hist_min + (loss_hist_max - loss_hist_min) * i / loss_hist_bins for i in range(loss_hist_bins + 1)]

    model.train()
    return {
        'overall': {
            'avg_loss': avg_loss,
            'perplexity': compute_perplexity(avg_loss),
            'accuracy': overall['sum_correct'] / overall['total_tokens'] if overall['total_tokens'] > 0 else 0.0,
            **topk,
            'repetition_rate': repetition_rate,
            'predictive_entropy': avg_entropy,
            'ece': ece,
            'ece_bins': ece_bins_out,
            'loss_hist_counts': overall['loss_hist_counts'],
            'loss_hist_edges': loss_hist_edges,
            'total_tokens': overall['total_tokens'],
        },
        'per_prompt': per_prompt,
    }
