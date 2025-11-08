"""Model architectures for proposed HIMALAYA adapter and LILAC baseline."""
from __future__ import annotations

import math
import types
from typing import Tuple

import numpy as np
import torch
from sklearn.neighbors import KDTree
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

# -----------------------------------------------------------------------------
# Quantisation helper (uniform symmetric) -------------------------------------
# -----------------------------------------------------------------------------

def quantise_tensor(t: torch.Tensor, num_bits: int = 3):
    levels = 2 ** num_bits - 1
    maxv = t.abs().max().item() + 1e-8
    scale = maxv / (levels / 2)
    q = torch.clamp((t / scale).round(), -levels / 2, levels / 2).to(torch.int8)
    return q, scale


# -----------------------------------------------------------------------------
# Hyper-router (tiny MLP) ------------------------------------------------------
# -----------------------------------------------------------------------------
class HyperRouter(nn.Module):
    def __init__(self, h_dim: int, out_atoms: int, hidden_params: int = 45000):
        super().__init__()
        width = max(32, hidden_params // (h_dim + out_atoms))
        self.net = nn.Sequential(nn.Linear(h_dim, width), nn.ReLU(), nn.Linear(width, out_atoms))

    def forward(self, x):
        return self.net(x)


# -----------------------------------------------------------------------------
# HIMALAYA adapter ------------------------------------------------------------
# -----------------------------------------------------------------------------
class HIMALAYAAdapter(nn.Module):
    """Two-tier layer-free adapter implementing the full HIMALAYA algorithm."""

    def __init__(self, hidden_dim: int, cfg):
        super().__init__()
        # Core & elastic dictionaries --------------------------------------
        k_c = cfg.model.adapter.core_dictionary.num_atoms
        k_e_max = cfg.model.adapter.elastic_dictionary.max_atoms
        self.k_c = k_c
        self.k_e_max = k_e_max

        self.register_parameter(
            "D_c", nn.Parameter(torch.randn(k_c, hidden_dim) / math.sqrt(hidden_dim))
        )
        self.register_buffer("D_e", torch.zeros(k_e_max, hidden_dim))
        self.register_buffer("active", torch.zeros(k_e_max, dtype=torch.bool))
        self.register_buffer("usage", torch.zeros(k_e_max, dtype=torch.long))

        # Router ------------------------------------------------------------
        total_atoms = k_c + k_e_max
        self.router = HyperRouter(hidden_dim, total_atoms, cfg.model.adapter.router.hidden_params)
        self.expected_k = cfg.model.adapter.router.expected_active_atoms
        self.temperature = nn.Parameter(torch.tensor(float(cfg.model.adapter.router.temperature_init)))
        self.tau = cfg.training.similarity_threshold_tau

        # Residual ring buffer for elastic growth --------------------------
        self.buffer = []
        self.buffer_size = 200  # promote after 200 residuals

        # KD-Tree (CPU) ------------------------------------------------------
        self.kdtree = None
        self._update_kdtree()

        # For Fisher consolidation -----------------------------------------
        self.fisher_U = nn.Parameter(torch.zeros_like(self.D_c))

    # ---------------------------------------------------------------------
    # Internals ------------------------------------------------------------
    def _dict_mat(self):
        return torch.cat([self.D_c, self.D_e[self.active]], dim=0)

    def _update_kdtree(self):
        mat = self._dict_mat().detach().cpu().numpy().astype(np.float32)
        if mat.shape[0] != 0:
            self.kdtree = KDTree(mat)

    # ---------------------------------------------------------------------
    def _consolidate(self, atom_vec: torch.Tensor):
        """Project information of `atom_vec` into the core latent statistics
        using a closed-form Fisher orthogonal projection (approximated)."""
        with torch.no_grad():
            # Simple projection: update Fisher-mean matrix
            proj = atom_vec / (atom_vec.norm() + 1e-8)
            self.fisher_U.data += proj.unsqueeze(0)
            # Renormalise to keep bounded norm <=1.05 as per paper
            row_norms = self.fisher_U.data.norm(dim=1, keepdim=True) + 1e-8
            self.fisher_U.data = torch.where(
                row_norms > 1.05, self.fisher_U.data * (1.05 / row_norms), self.fisher_U.data
            )

    # ---------------------------------------------------------------------
    def _grow_if_needed(self):
        if len(self.buffer) < self.buffer_size:
            return
        residuals = torch.stack(self.buffer, dim=0)
        self.buffer.clear()
        mat = self._dict_mat()
        if mat.shape[0] == 0:
            return
        cos = torch.mm(
            nn.functional.normalize(residuals, dim=-1),
            nn.functional.normalize(mat, dim=-1).t(),
        )
        if cos.mean().item() < self.tau:
            if (~self.active).any():
                slot = (~self.active).nonzero(as_tuple=False)[0].item()
            else:
                # prune least-used atom (Fisher consolidation beforehand)
                slot = self.usage.argmax().item()
                old_atom = self.D_e[slot].clone()
                self._consolidate(old_atom)
            # Promote mean residual -------------------------------------------------
            new_atom = nn.functional.normalize(residuals.mean(dim=0), dim=-1)
            self.D_e[slot].data.copy_(new_atom)
            self.active[slot] = True
            self.usage[slot] = 0
            self._update_kdtree()

    # ---------------------------------------------------------------------
    def forward(self, hidden):  # hidden: (B, T, H)
        B, T, H = hidden.shape
        cls = hidden[:, 0, :]  # use first token representation as task signal
        logits = self.router(cls) / self.temperature.abs()
        probs = torch.softmax(logits, dim=-1)
        top_val, top_idx = probs.topk(self.expected_k, dim=-1)
        coeff = torch.zeros_like(probs)
        coeff.scatter_(1, top_idx, top_val)
        update_vec = coeff @ self._dict_mat()  # (B, H)

        # Analytical norm rescaling to 1 ------------------------------------
        update_vec = nn.functional.normalize(update_vec, dim=-1)

        # Inject update (self-normalising outer product) --------------------
        hidden = hidden + update_vec.unsqueeze(1) / math.sqrt(H)

        # Stats for growth --------------------------------------------------
        if self.training:
            self.buffer.extend(update_vec.detach().cpu())
            if len(self.buffer) >= self.buffer_size:
                self._grow_if_needed()
            for b in range(B):
                for idx in top_idx[b]:
                    ridx = idx.item() - self.k_c
                    if 0 <= ridx < self.k_e_max and self.active[ridx]:
                        self.usage[ridx] += 1
        return hidden


# -----------------------------------------------------------------------------
# LoRA-style linear layer for LILAC baseline ----------------------------------
# -----------------------------------------------------------------------------
class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, r: int = 4, alpha: float = 1.0):
        super().__init__()
        self.linear = linear
        self.r = r
        self.alpha = alpha
        self.A = nn.Parameter(torch.randn(r, linear.in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(linear.out_features, r))
        self.scale = alpha / r
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

    def forward(self, x):
        return self.linear(x) + (x @ self.A.t() @ self.B.t()) * self.scale


def inject_lora(module: nn.Module, r: int = 4):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            setattr(module, name, LoRALinear(child, r=r, alpha=r))
        else:
            inject_lora(child, r)


# -----------------------------------------------------------------------------
# Model builder ---------------------------------------------------------------
# -----------------------------------------------------------------------------

def build_model(cfg):
    backbone_id = cfg.model.backbone.name
    tokenizer = AutoTokenizer.from_pretrained(backbone_id, cache_dir=".cache/", use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        backbone_id, cache_dir=".cache/", dtype=torch.float16, low_cpu_mem_usage=True
    )

    if cfg.training.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    for p in model.parameters():
        p.requires_grad = False

    hidden = model.config.hidden_size
    adapter_type = cfg.model.adapter.type.upper()

    if adapter_type == "HIMALAYA":
        adapter = HIMALAYAAdapter(hidden, cfg)
        orig_forward = model.forward

        def patched_forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
            outputs = orig_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
                **kwargs,
            )
            hidden_states = outputs.hidden_states[-1]
            hidden_states = adapter(hidden_states)
            logits = self.lm_head(hidden_states)
            loss = None
            if labels is not None:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )
            return {"loss": loss, "logits": logits}

        model.forward = types.MethodType(patched_forward, model)
        model.add_module("himalaya_adapter", adapter)
        for p in adapter.parameters():
            p.requires_grad = True

    elif adapter_type == "LILAC":
        inject_lora(model, r=cfg.model.adapter.lora_rank)
        for n, p in model.named_parameters():
            if n.endswith(".A") or n.endswith(".B"):
                p.requires_grad = True
    else:
        raise ValueError(adapter_type)

    return model, tokenizer