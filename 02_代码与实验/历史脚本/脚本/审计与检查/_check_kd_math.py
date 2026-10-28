"""Numeric check of the soft_kd_loss temperature bookkeeping.

GPT raised a real risk: if the cache already stores softmax(logits/T), then
training must use log_softmax(student_logits/T) and multiply by T^2 -- and must
NOT softmax the cached probabilities again.

This script checks the implementation against a hand-computed reference.
"""
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/jhy/rare_diag_20260915")
from _kd_train import soft_kd_loss

torch.manual_seed(0)
T = 2.0
w_min = 0.3
C, H, W = 8, 4, 4

# ground truth teacher logits, then the cache = softmax(ens / T)
teacher_logits = torch.randn(1, C, H, W) * 3.0
cache = torch.softmax(teacher_logits / T, dim=1)

student_logits = torch.randn(1, C, H, W) * 2.0

# ---- reference: plain weighted KL(q || p) * T^2, q = cache, p = softmax(s/T)
q = cache
log_p = F.log_softmax(student_logits / T, dim=1)
log_q = torch.log(q.clamp_min(1e-12))
kl_ref = (q * (log_q - log_p)).sum(dim=1)
pmax = q.max(dim=1).values
w = (w_min + (1 - w_min) * (pmax - 1.0 / C) / (1.0 - 1.0 / C)).clamp(w_min, 1.0)
ref = (w * kl_ref).mean() * T ** 2

got, wmean = soft_kd_loss(student_logits, cache, T, w_min)

print(f"reference = {ref.item():.8f}")
print(f"actual    = {got.item():.8f}")
print(f"abs diff  = {abs(ref.item() - got.item()):.3e}")
print(f"w mean    = {wmean:.4f}")

# ---- sanity: does the loss behave like a KL (>= 0, ~0 when student == teacher)?
same, _ = soft_kd_loss(teacher_logits, cache, T, w_min)
print(f"\nstudent == teacher logits -> loss = {same.item():.3e}  (expect ~0)")

# ---- catch the specific bug GPT warned about: softmax-ing the cache AGAIN
q_double = torch.softmax(cache, dim=1)
print(f"cache sums to 1? max|sum-1| = {(cache.sum(1).max() - 1).abs().item():.3e}")
print(f"softmax(cache) vs cache max diff = {(q_double - cache).abs().max().item():.3e}")
print("-> if the observed loss matched a double-softmax reference it would differ "
      "materially from the plain-KL reference above")
